import operator
import typing as t
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as nf
from networkx.utils import UnionFind
from torch import fx, nn

from ..utils import param_remove_

__all__ = ["Symbol", "PruneGroup", "SymbolicFilterInferer", "default_handlers"]


class Symbol(t.NamedTuple):
    node_name: str
    ch_idx: int

    def __repr__(self) -> str:
        return f"{self.node_name}/{self.ch_idx}"

    __str__ = __repr__

    @classmethod
    def recurse_expose_symbols(cls, value: object) -> t.Iterator["Symbol"]:
        if isinstance(value, Symbol):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from cls.recurse_expose_symbols(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from cls.recurse_expose_symbols(item)


class PruneGroup(t.NamedTuple):
    nodes: t.Tuple[str]
    filters: np.ndarray

    @property
    def n_filters(self):
        return self.filters.shape[1]

    def __repr__(self) -> str:
        return f"PruneGroup({self.nodes}, n_filters={self.n_filters})"

    __str__ = __repr__

    def __hash__(self) -> int:
        # There will not be 2 groups with the same nodes but different filters
        # so this should be fine.
        return hash(self.nodes)


HandlerTyT = t.Type["ShapeHandler"]
HandlerClassesT = t.Dict[t.Type, HandlerTyT]
HandlersT = t.Dict[t.Type, "ShapeHandler"]
# Types used in SymbolicFilterInferer
MaskT = t.List[t.Union[Symbol, bool]]
EqGroupT = t.List[t.Tuple[t.Union[Symbol, bool], t.Union[Symbol, bool]]]
# Interpretation value passed around in ShapeFixer
# First value is the concrete execution output
# Second value is the filters removed (empty if this layer's output is not 4D)
ShapeFixerT = t.Tuple[t.Any, t.List[int]]


class InfererAndFixer:
    def __init__(
        self, module: nn.Module, extra_handlers: HandlerClassesT = None
    ) -> None:
        self.module = fx.symbolic_trace(module)
        # User's handlers should overwrite ours in the cases of conflict
        self.handlers_cls: HandlerClassesT = {
            **default_handlers,
            **(extra_handlers or {}),
        }
        self.inferer = None
        self.fixer = None

    def infer_shape(self, *args):
        self.inferer = SymbolicFilterInferer(self.module, self.handlers_cls)
        self.inferer.run(*args)
        return self.inferer.groups

    def fix_shape(self, pruned_filter, *args):
        if not self.inferer:
            raise RuntimeError("Must call infer_shape() first")
        self.fixer = ShapeFixer(self.module, pruned_filter, self.inferer.handlers)
        self.fixer.run(*args)


class SymbolicFilterInferer(fx.Interpreter):
    def __init__(self, module: fx.GraphModule, handler_classes: HandlerClassesT):
        super().__init__(module)
        self.handler_classes = handler_classes
        self.handlers = {}
        self._variables = UnionFind()

    @property
    def variables(self) -> t.List[t.Set[Symbol]]:
        # Pop the set with `True` off the variables because they are not "variable" (or prunable) at all
        return [set_ for set_ in self._variables.to_sets() if True not in set_]

    @property
    def groups(self) -> t.List[PruneGroup]:
        from collections import defaultdict

        by_node_group = defaultdict(list)
        # Pop the set with `True` off the variables because they are not "variable" (or prunable) at all
        for set_ in self.variables:
            # These two are sorted
            set_list = list(sorted(set_, key=lambda x: x.node_name))
            node_group = tuple(node_name for node_name, _ in set_list)
            by_node_group[node_group].append([ch_idx for _, ch_idx in set_list])
        return [
            PruneGroup(nodes=node_group, filters=np.stack(filters).T)
            for node_group, filters in by_node_group.items()
        ]

    def output(self, target: str, args, kwargs) -> MaskT:
        # Here we simply set all arguments to True (meaning must-be-preserved -- unprunable)
        # because we don't want pruning to change the output shape
        (output,) = args
        for var in Symbol.recurse_expose_symbols(output):
            self._variables.union(var, True)
        return output

    def placeholder(self, target: str, args, kwargs) -> MaskT:
        # Here `args` are concrete values, because `args` is what user passed in
        # instead of something we returned
        concrete_ret = super().placeholder(target, args, kwargs)
        # If not 4D, no notion of filters.
        if len(concrete_ret.shape) != 4:
            return []
        # Generate a const mask of all Trues in channel (filter) dimension
        # because we can't prune filter here -- only after Conv layers.
        return [True for _ in range(concrete_ret.shape[1])]

    def call_module(self, target: str, args, kwargs) -> MaskT:
        module: nn.Module = self.submodules[target]
        return self._run_handler(target, type(module), module, args, kwargs)

    def call_function(self, target: t.Callable, args, kwargs) -> MaskT:
        return self._run_handler(target.__name__, target, target, args, kwargs)

    def _run_handler(
        self,
        target_name: str,
        target_ty: t.Union[t.Callable, t.Type],
        module_or_func: t.Union[nn.Module, t.Callable],
        args,
        kwargs,
    ) -> MaskT:
        handler_cls = self.handler_classes.get(target_ty)
        if handler_cls is None:
            # If no Symbols (prunable filter) input to this node at all, then it's fine.
            symbols = Symbol.recurse_expose_symbols(args)
            if not list(symbols):
                return []
            # Otherwise, we definitely need a handler.
            raise KeyError(f"Handler for {target_ty} not found")
        self.handlers[target_name] = handler = handler_cls(target_name, module_or_func)
        constraints, mask_ret = handler.get_mask(*args, **kwargs)
        # Put each variable into its own set
        for var in mask_ret:
            self._variables.union(var)
        # Then unify variables by the constraints
        for lhs, rhs in constraints:
            self._variables.union(lhs, rhs)
        return mask_ret


class ShapeFixer(fx.Interpreter):
    def __init__(
        self,
        module: fx.GraphModule,
        pruned_filters: t.Dict[str, t.List[int]],
        handlers: HandlersT,
    ) -> None:
        super().__init__(module)
        self.pruned_filters = pruned_filters
        self.handlers = handlers

    def placeholder(self, target: str, args, kwargs) -> ShapeFixerT:
        return super().placeholder(target, args, kwargs), []

    def call_module(self, target: str, args, kwargs) -> ShapeFixerT:
        return self._fix_shape(target, args, kwargs)

    def call_function(self, target: t.Callable, args, kwargs) -> ShapeFixerT:
        return self._fix_shape(target.__name__, args, kwargs)

    def _fix_shape(self, target_name: str, args, kwargs):
        con_output, ch_removed = self.handlers[target_name].fix_shape(*args, **kwargs)
        ch_removed = sorted(ch_removed + self.pruned_filters.get(target_name, []))
        return con_output, ch_removed


class ShapeHandler(ABC):
    def __init__(self, target_name: str, module_or_func) -> None:
        self.target_name = target_name
        self.callable = module_or_func

    @abstractmethod
    def get_mask(self, *args: MaskT, **kwargs) -> t.Tuple[EqGroupT, MaskT]:
        pass

    @abstractmethod
    def fix_shape(self, *args: ShapeFixerT, **kwargs) -> ShapeFixerT:
        pass


class Conv2dShapeHandler(ShapeHandler):
    callable: t.Union[nn.Conv2d, nn.ConvTranspose2d]

    def get_mask(self, x: MaskT) -> t.Tuple[EqGroupT, MaskT]:
        assert isinstance(self.callable, (nn.Conv2d, nn.ConvTranspose2d))
        if self.callable.dilation != (1, 1):
            raise ValueError("Dilation is not supported")
        groups = self.callable.groups
        if groups == 1:
            out_ch = self.callable.out_channels
            return [], [Symbol(self.target_name, i) for i in range(out_ch)]
        else:
            raise ValueError("Not supported yet")

    def fix_shape(self, x: ShapeFixerT) -> ShapeFixerT:
        con_x, ch_removed = x
        if isinstance(self.callable, nn.Conv2d):
            in_dim = 1
        elif isinstance(self.callable, nn.ConvTranspose2d):
            in_dim = 0
        else:
            raise ValueError(f"{self.callable} is not a supported convolution layer")
        self.callable.weight = param_remove_(self.callable.weight, in_dim, ch_removed)
        con_output = self.callable(con_x)
        self.callable.in_channels = self.callable.weight.shape[in_dim]
        return con_output, []


class BN2dShapeHandler(ShapeHandler):
    callable: nn.BatchNorm2d

    def get_mask(self, x: MaskT) -> t.Tuple[EqGroupT, MaskT]:
        return [], x

    def fix_shape(self, x: ShapeFixerT) -> ShapeFixerT:
        con_x, ch_removed = x
        for attr in ("weight", "bias", "running_mean", "running_var"):
            params = getattr(self.callable, attr)
            if params is not None:
                setattr(self.callable, attr, param_remove_(params, 0, ch_removed))
        con_output = self.callable(con_x)
        self.callable.num_features = con_output.shape[1]
        return con_output, ch_removed


class LinearShapeHandler(ShapeHandler):
    callable: nn.Linear

    def __init__(self, target_name: str, module_or_func) -> None:
        super().__init__(target_name, module_or_func)
        self.hw: int = None

    def get_mask(self, x: MaskT) -> t.Tuple[EqGroupT, MaskT]:
        if x:
            n_conv_ch = len(x)
            assert self.callable.in_features % n_conv_ch == 0
            self.hw = self.callable.in_features // n_conv_ch
        return [], []

    def fix_shape(self, x: ShapeFixerT) -> ShapeFixerT:
        con_x, ch_removed = x
        # ch_removed is the filters removed in prev conv layer (if any)
        if ch_removed:
            assert self.hw is not None
            in_features_removed = [
                c * self.hw + i for c in ch_removed for i in range(self.hw)
            ]
            self.callable.weight = param_remove_(
                self.callable.weight, 1, in_features_removed
            )
            self.callable.in_features = self.callable.weight.shape[1]
        con_output = self.callable(con_x)
        return con_output, []


class BinaryOpShapeHandler(ShapeHandler):
    def get_mask(self, lhs: MaskT, rhs: MaskT) -> t.Tuple[EqGroupT, MaskT]:
        # NOTE: assumes shape all equal; does not account for broadcasting
        assert len(lhs) == len(rhs)
        # filters in lhs must match filters in rhs
        return list(zip(lhs, rhs)), lhs

    def fix_shape(self, lhs: ShapeFixerT, rhs: ShapeFixerT) -> ShapeFixerT:
        assert lhs[1] == rhs[1]
        return self.callable(lhs[0], rhs[0]), lhs[1]


class ReductionShapeHandler(ShapeHandler):
    def get_mask(self, x: MaskT, dim: int, *args, **kwargs) -> t.Tuple[EqGroupT, MaskT]:
        if dim == 1:  # Filter dim
            return [], x  # Prune as you wish
        return [], []  # No pruning

    def fix_shape(self, x: ShapeFixerT, dim: int, *args, **kwargs) -> ShapeFixerT:
        return self.callable(x[0], *args, dim=dim, **kwargs), x[1]


class ConcatShapeHandler(ShapeHandler):
    def get_mask(self, inputs: t.List[MaskT], dim: int) -> t.Tuple[EqGroupT, MaskT]:
        # If not concat-ing along the channel dim,
        # all filters in all inputs must match along the channel dimension
        if dim != 1 and dim != -3:
            eq_constraints = []
            for prev, next_ in zip(inputs, inputs[1:]):
                assert len(prev) == len(next_)
                eq_constraints.extend(zip(prev, next_))
            return eq_constraints, inputs[0]
        # Otherwise, no constraints but the resulted output is simply the concatenation of all inputs
        return [], [x for xs in inputs for x in xs]

    def fix_shape(self, inputs: t.List[ShapeFixerT], dim: int) -> ShapeFixerT:
        import numpy as np

        input_tensors, input_filters = zip(*inputs)
        # If not concat-ing along the channel dim,
        # all filters in all inputs must match along the channel dimension
        if dim != 1 and dim != -3:
            return self.callable(input_tensors, dim=dim), input_filters[0]
        # Otherwise, need to concat all "removed filters" information
        accum_offset = 0
        all_pruned_ch = []
        for tensor, ch_removed in inputs:
            ch_removed = np.array(ch_removed) + accum_offset
            dim_len_before_prune = tensor.shape[dim] + len(ch_removed)
            accum_offset += dim_len_before_prune
            all_pruned_ch.extend(ch_removed.tolist())
        return self.callable(input_tensors, dim=dim), all_pruned_ch


class FlattenShapeHandler(ShapeHandler):
    def get_mask(
        self, x: MaskT, start_dim: int = 0, end_dim: int = -1
    ) -> t.Tuple[EqGroupT, MaskT]:
        if not x:
            return [(ch, True) for ch in x], []
        # This has to be 4D input
        if start_dim < 0:
            start_dim += 4
        if end_dim < 0:
            end_dim += 4
        if start_dim != 1 or end_dim != 3:
            # We only handle conv2d->linear flatten and this is not it
            return [(ch, True) for ch in x], []
        return [], x

    def fix_shape(
        self, x: ShapeFixerT, start_dim: int = 0, end_dim: int = -1
    ) -> ShapeFixerT:
        return self.callable(x[0], start_dim, end_dim), x[1]


class DefaultShapeHandler(ShapeHandler):
    def get_mask(self, x: MaskT, *args, **kwargs) -> t.Tuple[EqGroupT, MaskT]:
        return [], x  # No shape changes & no constraints

    def fix_shape(self, x: ShapeFixerT, *args, **kwargs) -> ShapeFixerT:
        return self.callable(x[0], *args, **kwargs), x[1]


# This is an (incomplete) list of layers that don't need handling.
# fmt: off
default_handling_layers = [
    # Pooling (layers, operators)
    nn.AvgPool2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
    nf.avg_pool2d, nf.max_pool2d, nf.adaptive_avg_pool2d, nf.adaptive_max_pool2d,
    # Elementwise (layers, operators)
    nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.ELU, nn.SELU, nn.SiLU, nn.PReLU, nn.Hardswish,
    nf.relu, nf.relu6, nf.leaky_relu, nf.elu, nf.selu, nf.silu, nf.prelu, nf.hardswish,
    # misc
    nn.Dropout, nf.interpolate
]
# fmt: on

# Only layers/functions that may output 4D tensor (tensor that has a notion of channel)
# are handled here
# Linear layer is a special case added because we don't want to miss the chance to
# prune the last Conv layer that goes into the Linear layer.
default_handlers = {
    **{
        # Use layer class (type) object as key
        nn.Conv2d: Conv2dShapeHandler,
        nn.ConvTranspose2d: Conv2dShapeHandler,
        nn.BatchNorm2d: BN2dShapeHandler,
        nn.Linear: LinearShapeHandler,
        # For functions, use function name as key
        # (why is this not torch.add? Because module tracing shows (+) as operator.add)
        operator.add: BinaryOpShapeHandler,
        torch.cat: ConcatShapeHandler,
        torch.flatten: FlattenShapeHandler,
        torch.argmax: ReductionShapeHandler,
    },
    **{k: DefaultShapeHandler for k in default_handling_layers},
}
