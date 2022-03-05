from typing import Dict, List

import torch
from torch.nn import Module
from torchvision.models import resnet18

from yapt.analyses import CFG
from yapt.struct_pruning import FixedPruneRatioPruner, StructuredPruner
from yapt.struct_pruning.pruning import l2_score_metric
from yapt.struct_pruning.symch import PruneGroup


def draw_resnet18():
    model = resnet18()
    dummy_input = torch.rand(1, 3, 224, 224)
    CFG(model, (dummy_input,)).prettify_and_draw("resnet18.png")


def prune_resnet18_fix_ratio():
    model = resnet18(pretrained=True)
    dummy_input = torch.rand(1, 3, 224, 224)
    # Prune 20% of the filters from each layer
    # This method affects the model in-place
    FixedPruneRatioPruner(0.2).prune_module_(model, (dummy_input,))
    print(model)


# --- Task code outline below ---


class OneLayerPruner(StructuredPruner):
    def __init__(self, layer_name: str, prune_ratio: float) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.prune_ratio = prune_ratio

    def select_filters(
        self, module: Module, groups: List[PruneGroup]
    ) -> Dict[str, List[int]]:
        # --- TODO: replace with your own code here ---
        # config =
        raise NotImplementedError()
        return self.get_prune_filters_from_config(module, config, l2_score_metric)


class FirstGroupPruner(StructuredPruner):
    def __init__(self, layer_name: str, prune_ratio: float) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.prune_ratio = prune_ratio

    def select_filters(
        self, module: Module, groups: List[PruneGroup]
    ) -> Dict[str, List[int]]:
        # --- TODO: replace with your own code here ---
        # Prune the layer specified by self.layer_name AND
        # any other layers that need to be collaterally pruned.
        # config =
        raise NotImplementedError()
        return self.get_prune_filters_from_config(module, config, l2_score_metric)


def prune_resnet18_layer():
    model = resnet18(pretrained=True)
    dummy_input = torch.rand(1, 3, 224, 224)
    # Prune only "layer4.1.conv1" Conv2d layer by ~20%
    OneLayerPruner("layer4.1.conv1", 0.2).prune_module_(model, (dummy_input,))
    print(model)


def prune_resnet18_first_group():
    model = resnet18(pretrained=True)
    dummy_input = torch.rand(1, 3, 224, 224)
    # Prune only the first layer group by ~20%
    FirstGroupPruner("conv1", 0.2).prune_module_(model, (dummy_input,))
    print(model)


if __name__ == "__main__":
    draw_resnet18()
    # prune_resnet18_fix_ratio()
    # prune_resnet18_layer()
    # prune_resnet18_first_group()
