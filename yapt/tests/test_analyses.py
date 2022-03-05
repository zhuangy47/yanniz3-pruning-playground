import unittest

import torch
from torchvision import models

from yapt.analyses import CFG, get_summary


class TestCFG(unittest.TestCase):
    def test_mobilenetv3(self):
        from tempfile import NamedTemporaryFile

        mobilenetv3 = models.mobilenet_v3_large()
        dummy_input = torch.rand(1, 3, 224, 224)
        cfg = CFG(mobilenetv3, (dummy_input,))
        # Total number of nodes
        self.assertEqual(len(cfg.nx_graph.nodes()), 189)
        # Total number of conv layers
        convs = cfg.select_module_if(lambda m: isinstance(m, torch.nn.Conv2d))
        self.assertEqual(len(convs), 62)
        # Check the output size of some certain layer (randomly hand-picked)
        shape = cfg.nx_graph.nodes["features.4.block.0.0"]["shape"]
        self.assertListEqual(list(shape), [1, 72, 56, 56])
        # Draw the CFG (not tested, as long as no error is thrown)
        filename = NamedTemporaryFile(suffix=".png").name
        cfg.prettify_and_draw(filename)
        print(f"Drew mobilenetv3 CFG to {filename}")


class TestSummary(unittest.TestCase):
    def test_mobilenetv3(self):
        mobilenetv3 = models.mobilenet_v3_large()
        dummy_input = torch.rand(1, 3, 224, 224)
        summary = get_summary(mobilenetv3, (dummy_input,))
        self.assertEqual(len(summary), 171)
        self.assertEqual(len(summary[summary["type"] == torch.nn.Conv2d]), 62)
        self.assertEqual(summary["params"].sum(), 5483032)
        self.assertEqual(summary["flops"].sum(), 245311840)


class TestArchSupport(unittest.TestCase):
    def setUp(self) -> None:
        from tempfile import mkdtemp
        from pathlib import Path

        self.dir_name = Path(mkdtemp())
        print(f"Created temporary directory: {self.dir_name}")

    def test_classification_nets(self):
        # fmt: off
        for family in [
            "alexnet", "resnet", "vgg", "squeezenet", "densenet",
            "googlenet", "mobilenet", "mnasnet", "shufflenetv2",
            "efficientnet", "regnet"
        ]:
        # fmt: on
            pymodule_or_ctor = getattr(models, family)
            if hasattr(pymodule_or_ctor, "__all__"):
                for model_name in pymodule_or_ctor.__all__:
                    if not model_name.islower():
                        continue
                    self._run_on_model(getattr(models, model_name)(), model_name)
            else:
                self._run_on_model(getattr(models, family)(), family)
        self._run_on_model(models.inception_v3(), "inception_v3", 299)

    def _run_on_model(self, model, model_name, input_size: int = 224):
        model = model.eval()
        dummy_input = torch.rand(1, 3, input_size, input_size)
        cfg = CFG(model, (dummy_input,))
        cfg.prettify_and_draw(self.dir_name / f"{model_name}.dot")
        summary = get_summary(model, (dummy_input,))
        summary.to_csv(self.dir_name / f"{model_name}.csv")


if __name__ == "__main__":
    unittest.main()
