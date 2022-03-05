import logging
from typing import Dict, List, NamedTuple

import opentuner as ot
import pytorch_lightning as pl
from torch.nn import Module
from torch.utils.data import DataLoader

from ..analyses import get_summary
from ..utils import sample_input
from .pruning import StructuredPruner, l2_score_metric
from .symch import PruneGroup

logger = logging.getLogger(__name__)
logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(logging.WARNING)


class AutotunedPruner(StructuredPruner):
    def __init__(
        self,
        n_iter: int,
        flops_ratio: float,
        tuning_ratio_diff: float,
        n_calibrate_images: int,
        accuracy_metric: str,
        opentuner_db: str = None,
        logging_config: dict = None,
        extra_handlers=None,
    ) -> None:
        super().__init__(extra_handlers)
        self.flops_ratio = flops_ratio
        self.tuning_ratio_diff = tuning_ratio_diff
        self.n_calibrate_images = n_calibrate_images
        self.accuracy_metric = accuracy_metric
        # TuningRunMain.__init__ initializes its own logger, so we'll override it and use ours
        if logging_config:
            ot.tuningrunmain.the_logging_config = logging_config
        self.ot_args = opentuner_default_args()
        if opentuner_db:
            self.ot_args.database = opentuner_db
        self.ot_args.test_limit = n_iter

        self.validator = pl.Trainer(gpus=1, logger=False, checkpoint_callback=False)
        self.args = None
        self.configs = []

    def prune_module_(self, module: Module, args=None, optimizers=None):
        if not isinstance(module, pl.LightningModule):
            raise TypeError(
                "`module` must be a LightningModule when using AutotunedPruner "
                "to be able to run calibration for BN layers."
            )
        if args is None:
            args = (sample_input(module),)
        self.args = args
        return super().prune_module_(module, args, optimizers)

    def select_filters(
        self, module: pl.LightningModule, groups: List[PruneGroup]
    ) -> Dict[str, List[int]]:
        from opentuner.tuningrunmain import TuningRunMain

        # 1. Run tuning
        measurer = PruningMeasurer(self.ot_args, self, module, groups)
        trm = TuningRunMain(measurer, self.ot_args)
        # A little bit of hack to get the _real_ progress when duplicated configs exist
        measurer.set_progress_getter(lambda: trm.search_driver.test_count)
        # This is where opentuner runs
        trm.main()
        # 2. Get the best config
        best_config = measurer.kept_configs[0]
        self.configs = measurer.kept_configs
        return best_config.strategy

    def make_pruned_copy(
        self, module: pl.LightningModule, config: Dict[str, List[int]]
    ):
        import copy

        module = copy.deepcopy(module)
        assert self.args is not None
        self._intern_prune_module_(module, self.args, config)
        return module

    def predict_accuracy_of_module(self, module: pl.LightningModule) -> float:
        # Calibrate on a small portion of training set and get accuracy on validation set.
        # (We're already inside torch.no_grad() here)
        n_im = 0
        module = module.cuda()
        loader = module.train_dataloader()
        assert isinstance(loader, DataLoader)  # Not a list of DataLoaders
        for batch in loader:
            module.training_step(module.transfer_batch_to_device(batch), 0)
            n_im += len(batch[0])
            if n_im >= self.n_calibrate_images:
                break
        # Validating:
        results = self.validator.validate(module, verbose=False)
        accuracy = results[0][self.accuracy_metric]
        return accuracy


class PruningMeasurer(ot.MeasurementInterface):
    def __init__(
        self,
        args,
        pruner: AutotunedPruner,
        module: pl.LightningModule,
        groups: List[PruneGroup],
    ) -> None:
        from opentuner.measurement.inputmanager import FixedInputManager
        from opentuner.search.objective import ThresholdAccuracyMinimizeTime
        from tqdm import tqdm

        self.pruner = pruner
        self.module = module
        self.groups = {f"group_{i}": group for i, group in enumerate(groups)}
        self.pbar = tqdm(total=args.test_limit, leave=False)
        self.flops_ratio = pruner.flops_ratio
        self.max_ratio_diff = pruner.tuning_ratio_diff
        self.kept_configs: List[Config] = []

        assert self.pruner.args is not None
        self.original_flops = get_summary(module, self.pruner.args)["flops"].sum()

        # Minus sign because we minimize "accuracy" (flops ratio difference)
        objective = ThresholdAccuracyMinimizeTime(-self.max_ratio_diff)
        input_manager = FixedInputManager(size=len(groups))
        super(PruningMeasurer, self).__init__(
            args, input_manager=input_manager, objective=objective
        )
        logger.info("Tuning for pruning strategy:")
        logger.info(
            "Keep (%.2f +- %.3f)%% of the FLOPs of original model (%d +- %d FLOPs)",
            self.flops_ratio * 100,
            self.max_ratio_diff * 100,
            self.original_flops,
            self.original_flops * self.max_ratio_diff,
        )
        logger.info(
            "%d parameters to be tuned in %d iterations", len(groups), args.test_limit
        )

    def set_progress_getter(self, getter):
        self.progress_getter = getter

    def manipulator(self):
        manipulator = ot.ConfigurationManipulator()
        for name in self.groups:
            manipulator.add_parameter(ot.FloatParameter(name, 0, 1))
        return manipulator

    def seed_configurations(self):
        # Rule of thumb, filter prune ratio is 1 - sqrt(flops_ratio).
        # We can start from here.
        estm_prune_ratio = 1 - self.flops_ratio ** 0.5
        return [{name: estm_prune_ratio for name in self.groups}]

    def run(self, desired_result, input, limit):
        iteration = self.progress_getter()
        cfg = desired_result.configuration.data
        translated_cfg = {
            self.groups[group_name]: ratio for group_name, ratio in cfg.items()
        }
        filters_to_prune = self.pruner.get_prune_filters_from_config(
            self.module, translated_cfg, l2_score_metric
        )
        module = self.pruner.make_pruned_copy(self.module, filters_to_prune)
        # Get flops
        assert self.pruner.args is not None
        summary = get_summary(module, self.pruner.args)
        flops = summary["flops"].sum()
        flops_ratio = flops / self.original_flops
        flops_diff = abs(flops_ratio - self.flops_ratio)
        if flops_diff >= self.max_ratio_diff:
            # No need to measure accuracy of this config if FLOPs is far from the target.
            result = ot.Result(time=0, accuracy=-flops_diff)
            logger.debug("Config %s excluded: flops_ratio = %.3f", cfg, flops_ratio)
        else:
            # Minus sign because we're maximizing accuracy
            pred_acc = self.pruner.predict_accuracy_of_module(module)
            result = ot.Result(time=-pred_acc, accuracy=-flops_diff)
            self.kept_configs.append(
                Config(iteration, flops_ratio, pred_acc, filters_to_prune)
            )
            logger.debug(
                "Config %s taken: flops_ratio = %.3f, accuracy = %.3f",
                cfg,
                flops_ratio,
                pred_acc,
            )
        self.pbar.update(iteration - self.pbar.n)
        return result

    def save_final_config(self, config):
        self.kept_configs = sorted(
            self.kept_configs, key=lambda x: x.pred_accuracy, reverse=True
        )
        logger.info("Selected %d configs", len(self.kept_configs))
        logger.info("Best prune ratio config: %s", config.data)
        self.pbar.close()


class Config(NamedTuple):
    iteration: int
    flops_ratio: float
    pred_accuracy: float
    strategy: Dict[str, List[int]]


def opentuner_default_args():
    from opentuner import default_argparser

    args = default_argparser().parse_args([])
    args.no_dups = True  # Don't print duplicated config warnings
    return args
