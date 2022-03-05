import abc
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Type, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import Conv2d, ConvTranspose2d, Module

from ..analyses import get_summary
from ..utils import param_remove_, rgetattr, sample_input
from . import symch

logger = logging.getLogger(__name__)
# Used for type annotations
Conv2dT = Union[Conv2d, ConvTranspose2d]
# Used for assertions
ConvTypes = (Conv2d, ConvTranspose2d)

# fmt: off
__all__ = [
    "StructuredPruningCallback",
    "StructuredPruner", "FixedPruneRatioPruner",
    "load_pruned_checkpoint"
]
# fmt: on


class StructuredPruningCallback(pl.Callback):
    def __init__(
        self,
        pruner: "StructuredPruner",
        every_n_epoch: int,
        onnx_output_names: Iterable[str] = ("output",),
        onnx_remove_init: bool = True,
        use_lr_rewind: bool = True,
        load_prev_best=True,
    ) -> None:
        self.pruner = pruner
        self.every_n_epoch = every_n_epoch
        self.onnx_output_names = onnx_output_names
        self.onnx_remove_init = onnx_remove_init
        self.use_lr_rewind = use_lr_rewind
        self.load_prev_best = load_prev_best

        self.prune_step = 0
        self.model_input = None
        self.model_saver = None
        self.initial_params_flops = None

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.model_input = sample_input(pl_module)
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                self.model_saver = callback
        if self.model_saver is None:
            logger.warning(
                "No ModelCheckpoint callback found in Trainer. "
                "Model will not be saved after pruning."
            )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epoch:
            return
        # This will try to reload the best model checkpoint
        # in the last pruning-retraining big step,
        # and log accuracy of that model
        # (if not reloaded then just accuracy of current model).
        self._load_best_and_log_acc(trainer, pl_module, False)
        model_before = deepcopy(pl_module)

        self.prune_step += 1  # Adding here because we start at 1
        # This needs to happen after _try_load_prev_best_ and prune_step += 1
        self._reset_model_ckpt()
        logger.info("Pruning step %d invoked before epoch %d", self.prune_step, epoch)
        # Start this round of pruning
        pl_module = pl_module.eval()
        prune_decisions = self.pruner.prune_module_(
            pl_module,
            (self.model_input,),
            optimizers=trainer.optimizers,
        )
        if self.use_lr_rewind:
            # Learning rate rewinding!
            # Reinitialize optimizer and LR scheduler
            trainer.accelerator.setup_optimizers(trainer)
        # Print some stats
        self._log_prune_stats(trainer, model_before, pl_module, prune_decisions)
        pl_module.train()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Check if training is ending.
        # We do this instead of using on_train_end because logging is not
        # supported there.
        if trainer.current_epoch == trainer.max_epochs - 1:
            # Finalize: load best model and log accuracy
            self._load_best_and_log_acc(trainer, pl_module, True)

    def _load_best_and_log_acc(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, end_of_epoch: bool
    ):
        if self.load_prev_best and self.model_saver:
            best_model = self.model_saver.best_model_path
            # best_model will be None if e.g. epoch=0
            if best_model:
                logger.info(
                    "Loading from best checkpoint of previous pruning step: %s",
                    self.model_saver.best_model_path,
                )
                pl_module.load_state_dict(torch.load(best_model)["state_dict"])
        if trainer.logger:
            metrics = self._evaluate_model(pl_module)
            metrics = {f"pruned_metrics/{k}": v for k, v in metrics.items()}
            current_epoch = trainer.current_epoch + (1 if end_of_epoch else 0)
            trainer.logger.log_metrics(metrics, step=current_epoch)

    def _reset_model_ckpt(self):
        if self.model_saver is None:
            return
        saver: ModelCheckpoint = self.model_saver
        # Prefix checkpoint names with e.g. "prune_steps=0-"
        if saver.filename.startswith("prune_step"):
            filename = saver.filename.split("-", 1)[1]
        else:
            filename = saver.filename
        saver.filename = f"prune_step={self.prune_step}-" + filename
        # Force reset ModelCheckpoint history
        saver._last_global_step_saved = -1
        saver.current_score = None
        saver.best_k_models = {}
        saver.kth_best_model_path = ""
        saver.best_model_score = None
        saver.best_model_path = ""
        saver.last_model_path = ""

    def _log_prune_stats(
        self,
        trainer: pl.Trainer,
        model_before: pl.LightningModule,
        model_after: pl.LightningModule,
        prune_decisions: Dict[str, List[int]],
    ):
        params_before, flops_before = self._get_params_flops(model_before)
        if self.initial_params_flops is None:
            self.initial_params_flops = params_before, flops_before
        params_after, flops_after = self._get_params_flops(model_after)
        params_0, flops_0 = self.initial_params_flops
        params_rate, flops_rate = (
            params_before / params_after,
            flops_before / flops_after,
        )
        params_rate_0, flops_rate_0 = params_0 / params_after, flops_0 / flops_after
        logger.info(
            "%.3fX reduction in parameters after pruning (%d -> %d) (%.3fX accumulative)",
            params_rate,
            params_before,
            params_after,
            params_rate_0,
        )
        logger.info(
            "%.3fX reduction in FLOPs after pruning (%d -> %d) (%.3fX accumulative)",
            flops_rate,
            flops_before,
            flops_after,
            flops_rate_0,
        )
        if trainer.logger:
            trainer.logger.log_metrics(
                {
                    "prune_ratio/params": params_rate_0,
                    "prune_ratio/flops": flops_rate_0,
                },
                step=trainer.current_epoch,
            )

        lines = []
        for layer_name, pruned_filters in prune_decisions.items():
            nfilter_before = rgetattr(model_before, layer_name).weight.shape[0]
            lines.append(
                f"  {layer_name}: {len(pruned_filters)} pruned out of {nfilter_before}"
            )
            lines.append(f"  filters: {pruned_filters}")
        logger.debug("Layer pruned: \n%s", "\n".join(lines))

    def _get_params_flops(self, module: pl.LightningModule):
        summary = get_summary(module, (self.model_input,))
        return summary["params"].sum(), summary["flops"].sum()

    @staticmethod
    @torch.no_grad()
    def _evaluate_model(pl_module: pl.LightningModule):
        from pytorch_lightning.core.step_result import Result
        from pytorch_lightning.utilities.apply_func import move_data_to_device
        from tqdm import tqdm

        # This used to create a new Trainer and invoke it
        # but that seems to create some internal invisible state
        # that makes backprop after pruning fail.
        # We're now doing it manually. Please don't go back to that again.
        pl_module = pl_module.eval()
        dl = pl_module.val_dataloader()
        results = []
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl)):
            batch = move_data_to_device(batch, pl_module.device)
            pl_module.validation_step(batch, batch_idx)
            results.append(pl_module._results)
            pl_module._results = Result()
        metrics = dict(Result.reduce_on_epoch_end(results))
        metrics.pop("meta")
        pl_module = pl_module.train()
        return metrics


ScoreMetricT = Callable[[List[torch.Tensor], List[int]], float]


class StructuredPruner(abc.ABC):
    def __init__(self, extra_handlers=None) -> None:
        super().__init__()
        self.extra_handlers = extra_handlers

    @abc.abstractmethod
    def select_filters(
        self, module: Module, groups: List[symch.PruneGroup]
    ) -> Dict[str, List[int]]:
        pass

    @torch.no_grad()
    def prune_module_(self, module: Module, args, optimizers=None):
        # Run shape inference on the module to get groups to prune.
        inf_fix = symch.InfererAndFixer(module, extra_handlers=self.extra_handlers)
        groups = inf_fix.infer_shape(*args)
        chans_to_prune = self.select_filters(module, groups)
        logger.info("Filters to be removed: %s", chans_to_prune)
        # Perform actual pruning
        self._intern_prune_module_(module, args, chans_to_prune)
        # Clear tensor states of optimizers (because the shape may mismatch)
        # TODO: it's apparently not the best to remove all optimizer states.
        # Better would be to drop filters for each state tensor in each optimizer
        # just like how the weights are pruned.
        optimizers = optimizers or []
        for optim in optimizers:
            for param in optim.state.keys():
                optim.state[param] = {}
        return chans_to_prune

    def _intern_prune_module_(
        self, module: Module, args, chans_to_prune: Dict[str, List[int]]
    ):
        # 1. Run shape inference on the module, just so that we can run `fix_shape` later.
        inf_fix = symch.InfererAndFixer(module, extra_handlers=self.extra_handlers)
        inf_fix.infer_shape(*args)
        # 2. Prune each conv layer
        for layer_name, chans in chans_to_prune.items():
            layer = rgetattr(module, layer_name)
            assert isinstance(layer, ConvTypes)
            _prune_conv2d_(layer, chans)  # inplace
        # Fix layers affected by shape changes
        inf_fix.fix_shape(chans_to_prune, *args)
        # 3. Validate model by just running the inference once
        module(*args)

    @classmethod
    def get_prune_filters_from_config(
        cls,
        module: Module,
        prune_cfg: Dict[symch.PruneGroup, float],
        filters_score_metric: ScoreMetricT,
    ) -> Dict[str, List[int]]:
        ret = {}
        for group, ratio in prune_cfg.items():
            ret.update(
                cls._get_prune_filters_of_group(
                    module, group, ratio, filters_score_metric
                )
            )
        return ret

    @classmethod
    def _get_prune_filters_of_group(
        cls,
        module: Module,
        group: symch.PruneGroup,
        prune_ratio: float,
        filters_score_metric: ScoreMetricT,
    ) -> Dict[str, List[int]]:
        import math

        import numpy as np

        nch = group.n_filters
        # Rank each "filter group" (1 filter per layer from the group) by criteria
        conv_weights = [
            rgetattr(module, layer_name).weight for layer_name in group.nodes
        ]
        scores = [
            filters_score_metric(conv_weights, filter_group)
            for filter_group in group.filters.T
        ]  # Of length `nch`
        # Sort filter groups by score, and take the top `nch_to_remove`
        # with the LOWEST scores.
        # Always keep at least one filter
        nch_to_remove = min(math.ceil(nch * prune_ratio), nch - 1)
        # filters_to_remove: n_layers x nch_to_remove
        filters_to_remove = group.filters.T[
            np.argsort(np.array(scores))[:nch_to_remove]
        ].T
        return {
            layer_name: filters_to_remove[i].tolist()
            for i, layer_name in enumerate(group.nodes)
        }


def l2_score_metric(weights: List[torch.Tensor], filter_indices: List[int]) -> float:
    """Returns sum of l2-norm of filter "normalized" by l2-norm of weight"""

    norm = lambda x: torch.norm(x.flatten(), p=2).item()
    return sum(
        [
            norm(weight[filter_idx]) / norm(weight)
            for weight, filter_idx in zip(weights, filter_indices)
        ]
    )


class FixedPruneRatioPruner(StructuredPruner):
    def __init__(self, prune_ratio: float, extra_handlers=None) -> None:
        super().__init__(extra_handlers)
        self.prune_ratio = prune_ratio

    def select_filters(
        self, module: Module, groups: List[symch.PruneGroup]
    ) -> Dict[str, List[int]]:
        config = {group: self.prune_ratio for group in groups}
        return self.get_prune_filters_from_config(module, config, l2_score_metric)


def load_pruned_checkpoint(
    cls: Type[pl.LightningModule],
    checkpoint_path,
    map_location=None,
    hparams_file=None,
    strict: bool = True,
    extra_handlers=None,
    **kwargs,
):
    def on_load_checkpoint(self: pl.LightningModule, checkpoint: Dict[str, Any]):
        filters_pruned = {}
        state_dict = checkpoint["state_dict"]
        inf_fix = symch.InfererAndFixer(self, extra_handlers=extra_handlers)
        args = self.example_input_array
        if args is None:
            raise ValueError("Module must define `example_input_array`.")
        inf_fix.infer_shape(args)
        for name, module in self.named_modules():
            if not isinstance(module, ConvTypes):
                continue
            weight_name = f"{name}.weight"
            if weight_name not in state_dict:
                continue
            n_filters = state_dict[weight_name].size(0)
            n_removed = module.weight.size(0) - n_filters
            # Do a dummy pruning that removes the first n filters
            # so that the shape matches. We'll load in the actual weights soon anyway.
            filters_pruned[name] = list(range(n_removed))
        for layer_name, removed_chans in filters_pruned.items():
            layer = rgetattr(self, layer_name)
            assert isinstance(layer, ConvTypes)
            _prune_conv2d_(layer, removed_chans)
        inf_fix.fix_shape(filters_pruned, args)
        # Validate model by just running the inference once
        self(args)

    __on_load_checkpoint = cls.on_load_checkpoint
    cls.on_load_checkpoint = on_load_checkpoint
    ret = cls.load_from_checkpoint(
        checkpoint_path, map_location, hparams_file, strict, **kwargs
    )
    cls.on_load_checkpoint = __on_load_checkpoint
    return ret


def _prune_conv2d_(layer: Conv2dT, prune_filters: List[int]):
    # This only remove the output filters.
    # The input filters are later fixed along with number of filters of BatchNorm, Linear...
    # as that's a data dependency problem.
    layer.out_channels -= len(prune_filters)
    if isinstance(layer, Conv2d):
        out_dim = 0
    elif isinstance(layer, ConvTranspose2d):
        out_dim = 1
    else:
        raise ValueError(f"Unknown type of layer {layer}")
    layer.weight = param_remove_(layer.weight, out_dim, prune_filters)
    if layer.bias is not None:
        layer.bias = param_remove_(layer.bias, 0, prune_filters)
