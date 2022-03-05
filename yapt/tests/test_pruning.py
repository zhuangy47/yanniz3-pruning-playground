import math
import unittest

import pytorch_lightning as pl
import torch
from torch import fx
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from yapt.struct_pruning import FixedPruneRatioPruner, AutotunedPruner
from yapt.utils import rgetattr


class ModelPLWrapper(pl.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val/loss", loss)
        self.log("val/accuracy", accuracy)
        return accuracy

    def train_dataloader(self):
        dummy_dataset = TensorDataset(
            torch.randn(100, 3, 224, 224), torch.randint(low=0, high=1000, size=(100,))
        )
        return DataLoader(dummy_dataset, num_workers=16)

    def val_dataloader(self):
        return self.train_dataloader()

    @property
    def example_input_array(self):
        return torch.rand(1, 3, 224, 224)

    def configure_optimizers(self):
        return {
            "optimizer": torch.optim.SGD(
                self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
            )
        }


class TestPruning(unittest.TestCase):
    def setUp(self) -> None:
        from torchvision.models.resnet import resnet18

        self.model = ModelPLWrapper(resnet18())
        self.dummy_input = torch.rand(1, 3, 224, 224)

    def test_symch(self):
        from collections import Counter

        from yapt.struct_pruning import SymbolicFilterInferer, default_handlers

        inferer = SymbolicFilterInferer(
            fx.symbolic_trace(self.model), default_handlers
        )
        inferer.run(self.dummy_input)

        all_layer_names = set(
            [var.node_name for set_ in inferer.variables for var in set_]
        )
        for layer_name in all_layer_names:
            self.assertIsInstance(rgetattr(self.model, layer_name), torch.nn.Conv2d)
        # Should have 1920 independent filters and 448 3-groups
        counter = Counter([len(set_) for set_ in inferer.variables])
        self.assertEqual(counter[1], 1920)
        self.assertEqual(counter[3], 960)

    def test_fixed_pruning(self):
        n_filters = get_n_filters(self.model)
        prune_decisions = FixedPruneRatioPruner(0.2).prune_module_(
            self.model, (self.dummy_input,)
        )
        for k, v in prune_decisions.items():
            self.assertEqual(len(v), math.ceil(n_filters[k] * 0.2))

    def test_tuning_pruning(self):
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile() as f:
            AutotunedPruner(
                10, 0.75, 0.01, 100, "val/accuracy", opentuner_db=f.name
            ).prune_module_(self.model)

    def test_pruning_iterative(self):
        from yapt.struct_pruning import StructuredPruningCallback

        N, R = 5, 0.2
        n_filters_before = get_n_filters(self.model)
        pruner = StructuredPruningCallback(FixedPruneRatioPruner(R), 1)
        trainer = pl.Trainer(
            logger=False,
            callbacks=[pruner],
            max_epochs=N,
            checkpoint_callback=False,
        )
        trainer.fit(self.model)
        n_filters_after = get_n_filters(self.model)
        for k, v in self.model.named_modules():
            if not isinstance(v, torch.nn.Conv2d):
                continue
            before = n_filters_before[k]
            for _ in range(N):
                before -= math.ceil(before * R)
            self.assertEqual(before, n_filters_after[k])


class TestArchSupport(unittest.TestCase):
    def test_alexnet(self):
        self._run_on_model(models.alexnet())

    def test_resnets(self):
        model_names = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
        # Unsupported due to having grouped convolutions:
        # "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2"
        self._run_on_models(model_names)

    def test_vggs(self):
        self._run_on_models(models.vgg.__all__)

    def test_squeezenets(self):
        self._run_on_models(models.squeezenet.__all__)

    def test_inception(self):
        self._run_on_model(models.inception_v3(init_weights=True), 299)

    def test_densenets(self):
        self._run_on_models(models.densenet.__all__)

    def test_googlenet(self):
        self._run_on_model(models.googlenet(init_weights=True))

    # # Not supported due to having grouped convolutions
    # def test_mobilenets(self):
    #     self._run_on_models(models.mobilenet.__all__)

    # # Not supported due to having grouped convolutions
    # def test_mnasnets(self):
    #     self._run_on_models(models.mnasnet.__all__)

    # # Not supported due to having grouped convolutions
    # def test_shufflenets(self):
    #     self._run_on_models(models.shufflenetv2.__all__)

    # # Not supported due to having grouped convolutions
    # def test_efficientnets(self):
    #     self._run_on_models(models.efficientnet.__all__)

    # # Not supported due to having grouped convolutions
    # def test_regnets(self):
    #     self._run_on_models(models.regnet.__all__)

    def _run_on_models(self, model_names, input_size: int = 224):
        for model in model_names:
            if model.islower():
                self._run_on_model(getattr(models, model)(), input_size)

    def _run_on_model(self, model, input_size: int = 224):
        model = model.eval()
        dummy_input = torch.rand(1, 3, input_size, input_size)
        n_filters = get_n_filters(model)
        prune_decisions = FixedPruneRatioPruner(0.2).prune_module_(
            model, (dummy_input,)
        )
        for k, v in prune_decisions.items():
            self.assertEqual(len(v), math.ceil(n_filters[k] * 0.2))


def get_n_filters(model):
    return {
        k: v.weight.shape[0]
        for k, v in model.named_modules()
        if isinstance(v, torch.nn.Conv2d)
    }


if __name__ == "__main__":
    unittest.main()
