import typing as t
from pathlib import Path

import networkx as nx
import torch
from torch.fx import node as fxnode
from torch.fx import symbolic_trace
from torch.nn import Module

from ..utils import rgetattr


class CFG(t.Mapping):
    """Control flow graph of a Torch Module.

    Internally the CFG is stored as a NetworkX MultiDiGraph, where each node is a string.
    """

    pos2color = {
        "call_module": "black",  # calling a module is the "plain" operation
        "placeholder": "darkgreen",
        "output": "darkgreen",  # placeholder/output are distinguished by arrow direction
        "call_function": "brown",
        "call_method": "darkgoldenrod",
        "get_attr": "cornflowerblue",
    }

    def __init__(self, module: torch.nn.Module, args: t.Any = None) -> None:
        from torch.fx.passes.shape_prop import ShapeProp

        self.graphmodule = symbolic_trace(module)
        if args is not None:
            # This will add the shape and dtype of each node's output
            # to the node, inplace.
            ShapeProp(self.graphmodule).run(*args)
        self.nx_graph = nx.MultiDiGraph()
        for fx_node in self.graphmodule.graph.nodes:
            fx_node: fxnode.Node
            for i, arg in enumerate(fx_node.args):
                self._add_edge(arg, fx_node, i)
            for kw, arg in fx_node.kwargs.items():
                self._add_edge(arg, fx_node, kw)
        # Make another pass to make sure every node has the same
        # attributes (even if the value is None)
        for _, attrs in self.nx_graph.nodes(data=True):
            attrs: dict
            fx_node = attrs["node"]
            attrs["shape"] = self._extract_shape_from_node(fx_node)
            if fx_node.op == "call_module":
                submodule = rgetattr(module, fx_node.target)
                attrs["module"] = submodule
                attrs["type"] = type(submodule)
            elif fx_node.op == "call_function":
                attrs["module"] = None
                attrs["type"] = fx_node.target  # `target` is the function itself
            else:
                attrs["module"] = attrs["type"] = None

    def _extract_shape_from_node(self, fx_node: fxnode.Node):
        from torch.fx.passes.shape_prop import TensorMetadata

        # ShapeProp adds results to fx_node.shape before torch 1.10
        if hasattr(fx_node, "shape"):
            return getattr(fx_node, "shape")
        # Starting from PyTorch 1.10 there is fx_node.meta
        if hasattr(fx_node, "meta"):
            tmeta = fx_node.meta.get("tensor_meta", None)
            if isinstance(tmeta, TensorMetadata):
                return tmeta.shape
        return None

    def _add_edge(
        self, from_: fxnode.Argument, to_: fxnode.Node, index: t.Union[int, str]
    ):
        stringify = lambda node: str(node).replace("_", ".")
        if isinstance(from_, fxnode.Node):
            # Add these fx-nodes by name so it's easier to look up
            # but preserve the actual node in the "node" attribute.
            self.nx_graph.add_node(stringify(from_), node=from_)
            self.nx_graph.add_node(stringify(to_), node=to_)
            self.nx_graph.add_edge(stringify(from_), stringify(to_), index=index)
        elif isinstance(from_, (tuple, list)):  # Can be a list of Nodes
            for i, f in enumerate(from_):
                self._add_edge(f, to_, f"{index}/{i}")
        elif isinstance(from_, dict):  # Can be a dict of Nodes
            for k, f in from_.items():
                self._add_edge(f, to_, f"{index}/{k}")
        # Intentionally not handling other types of arguments

    def sorted_inputs(self, node: t.Union[str, fxnode.Node]):
        from operator import itemgetter

        if isinstance(node, fxnode.Node):
            node = str(node)
        in_edges = self.nx_graph.in_edges(node, "index")
        sorted_edges = sorted(in_edges, key=itemgetter(2))
        return [e[0] for e in sorted_edges]

    def prettify_and_draw(self, to_file, dot_format: str = None):
        import subprocess
        from tempfile import NamedTemporaryFile

        from networkx.drawing.nx_pydot import write_dot

        to_file = Path(to_file)
        dot_format = dot_format or to_file.suffix.lstrip(".")
        drawable_graph = nx.MultiDiGraph()
        # Add node to new graph
        for node, attrs in self.nx_graph.nodes(data=True):
            attrs: dict
            fx_node = attrs["node"]
            node_name = f"{node}*" if attrs["module"] is None else node
            if fx_node.op == "call_module":
                type_str = attrs["type"].__name__
                label = f"{node_name}\\n{type_str}"
            else:
                label = node_name
            drawable_graph.add_node(
                node,
                shape="rectangle",
                style="rounded",
                color=self.pos2color[fx_node.op],
                label=label,
            )
        # Add shape as edge annotation
        for src, dst, _ in self.nx_graph.edges(keys=True):
            shape = self.nx_graph.nodes[src]["shape"]
            label = "" if shape is None else "x".join(str(x) for x in shape)
            drawable_graph.add_edge(src, dst, label=label)
        if dot_format == "dot":
            write_dot(drawable_graph, to_file)
        else:
            with NamedTemporaryFile("w") as tmp:
                write_dot(drawable_graph, tmp.name)
                subprocess.run(
                    ["dot", tmp.name, f"-T{dot_format}", "-o", to_file.as_posix()],
                    check=True,
                )

    def select_module_if(self, predicate: t.Callable[[Module], bool]) -> t.List[Module]:
        return [
            module
            for _, module in self.nx_graph.nodes(data="module")
            if predicate(module)
        ]

    def __getitem__(self, k: str) -> t.Optional[Module]:
        return self.nx_graph[k]["module"]

    def __iter__(self) -> t.Iterator[str]:
        return iter(self.nx_graph)

    def __len__(self) -> int:
        return len(self.nx_graph)
