from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import pandas
import torch
import torch.nn as nn
from torch import fx

HandlerSubclass = Callable[..., "ModuleHandler"]
HandlersDictT = Dict[Type, HandlerSubclass]


def get_summary(
    model: nn.Module, model_args: Tuple, extra_type_handlers: HandlersDictT = None
) -> pandas.DataFrame:
    graph_model = fx.symbolic_trace(model)
    return RunModelSummary(graph_model, extra_type_handlers).run(*model_args)


class RunModelSummary(fx.Interpreter):
    def __init__(
        self, module: fx.GraphModule, extra_type_handlers: HandlersDictT = None
    ):
        super().__init__(module)
        # User's handlers should overwrite ours in the cases of conflict
        self.handlers: HandlersDictT = {
            **default_type_handlers,
            **(extra_type_handlers or {}),
        }
        self.summary = {}

    def run(self, *args, initial_env: Optional[Dict[fx.Node, Any]] = None) -> Any:
        super().run(*args, initial_env=initial_env)
        return pandas.DataFrame(self.summary).T

    def call_function(self, target, args, kwargs):
        concrete_result = super().call_function(target, args, kwargs)
        self.summary[target.__name__] = summary = {
            "type": target.__name__,
            "params": 0,
            "trainable": False,
            "flops": 0,
        }
        self._apply_handler_(target, summary, args, concrete_result)
        return concrete_result

    def call_module(self, target: str, args, kwargs) -> Any:
        concrete_result = super().call_module(target, args, kwargs)
        module = self.submodules[target]
        n_params = sum(param.numel() for param in module.parameters())
        trainable = any(param.requires_grad for param in module.parameters())
        self.summary[target] = summary = {
            "type": type(module),  # Actual type
            "params": n_params,
            "trainable": trainable,
            "flops": 0,
        }
        self._apply_handler_(module, summary, args, concrete_result)
        return concrete_result

    def _apply_handler_(self, target, summary: dict, args, concrete_result):
        # Additional info on input/output shapes, and FLOPs
        handler_cls = self.handlers.get(summary["type"])
        if handler_cls is None:
            return
        try:
            handler = handler_cls(target, args, concrete_result)
            summary["input_shape"] = handler.input_shape
            summary["output_shape"] = handler.output_shape
            summary["flops"] = handler.flops
        except ValueError:
            return


class ModuleHandler(ABC):
    def __init__(self, target, inputs: Tuple, output: Any) -> None:
        self.target = target
        self.inputs = inputs
        self.output = output

    @property
    @abstractmethod
    def flops(self) -> float:
        pass

    @property
    @abstractmethod
    def input_shape(self) -> Any:
        pass

    @property
    @abstractmethod
    def output_shape(self) -> Any:
        pass


class DefaultSizeHandler(ModuleHandler, ABC):
    def __init__(self, target, inputs: Tuple, output: Any) -> None:
        super().__init__(target, inputs, output)
        self._input0_shape = self.default_sizes(inputs[0])
        self._output_shape = self.default_sizes(output)

    @staticmethod
    def default_sizes(value) -> List[int]:
        """Can throw ValueError"""
        try:
            if isinstance(value, torch.Tensor):
                return list(value.size())
        except AttributeError as e:
            raise ValueError(*e.args)
        raise ValueError(f"Size of value (type {type(value)}) not handle-able")

    @property
    def input_shape(self):
        return self._input0_shape

    @property
    def output_shape(self):
        return self._output_shape


class Pool2DHandler(DefaultSizeHandler):
    target: Union[nn.AvgPool2d, nn.MaxPool2d]

    @property
    def flops(self):
        ksize = self.target.kernel_size
        if isinstance(ksize, int):
            ksize = ksize, ksize
        k_area = ksize[0] * ksize[1]
        return k_area * _get_numel(self._output_shape)


class LinearHandler(DefaultSizeHandler):
    target: nn.Linear

    @property
    def flops(self):
        m, n = self._input0_shape
        k, n_ = self.target.weight.shape
        assert n == n_
        return m * n * k


class Conv2dHandler(DefaultSizeHandler):
    target: nn.Conv2d

    @property
    def flops(self):
        _, _, h, w = self._output_shape
        return self.target.weight.numel() * h * w


class BatchNorm2dHandler(DefaultSizeHandler):
    target: nn.BatchNorm2d

    @property
    def flops(self):
        return 6 * _get_numel(self._output_shape)


class InplaceHandler(DefaultSizeHandler):
    @property
    def flops(self):
        return _get_numel(self._output_shape)


default_type_handlers = {
    # Use layer class (type) object as key
    nn.Linear: LinearHandler,
    nn.Conv2d: Conv2dHandler,
    nn.BatchNorm2d: BatchNorm2dHandler,
    nn.ReLU: InplaceHandler,
    nn.ReLU6: InplaceHandler,
    nn.LeakyReLU: InplaceHandler,
    nn.AvgPool2d: Pool2DHandler,
    nn.MaxPool2d: Pool2DHandler,
    # For functions, use function name as key
    "add": InplaceHandler,
}


def _get_numel(shape):
    return torch.prod(torch.tensor(shape)).item()
