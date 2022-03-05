import functools
from pathlib import Path
from typing import Iterable, Tuple, Union

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor
from torch.nn import Module

PathLike = Union[str, Path]


# Taken from https://stackoverflow.com/questions/31174295
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)


def param_remove_(param: Tensor, dim: int, indices: Iterable[int]):
    remove_indices = set(indices)
    selected_indices = torch.tensor(
        [i for i in range(param.shape[dim]) if i not in remove_indices]
    ).to(param.device)
    param.data = torch.index_select(param.data, dim, selected_indices)
    if param.grad is not None:
        param.grad = torch.index_select(param.grad, dim, selected_indices)
    return param


def sample_input(pl_module: LightningModule):
    input_ = pl_module.example_input_array
    if input_ is None:
        raise ValueError("Module must define `example_input_array`.")
    return pl_module._apply_batch_transfer_handler(input_)


def resolve_trainer_dir(trainer: Trainer):
    from pytorch_lightning.loggers import TensorBoardLogger

    if not isinstance(trainer.logger, TensorBoardLogger):
        return Path(trainer.default_root_dir)
    return Path(trainer.logger.log_dir)


def torch_to_onnx(
    model: Module,
    model_args: Tuple,
    output_file: PathLike,
    output_names: tuple = ("output",),
    remove_init: bool = True,
):
    import onnx
    import onnxsim
    from torch.onnx import export

    def remove_initializer_from_input(model: onnx.ModelProto) -> onnx.ModelProto:
        inputs = model.graph.input
        name_to_input = {}
        for input in inputs:
            name_to_input[input.name] = input
        for initializer in model.graph.initializer:
            if initializer.name in name_to_input:
                inputs.remove(name_to_input[initializer.name])
        return model

    output_file = Path(output_file).as_posix()
    training = model.training
    if hasattr(model, "_onnx_exporting"):
        model._onnx_exporting = True
    export(
        model.eval(),
        model_args,
        output_file,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=output_names,  # the model's output names
        strip_doc_string=False,
        keep_initializers_as_inputs=True,
    )
    model.train(training)
    if hasattr(model, "_onnx_exporting"):
        model._onnx_exporting = False

    # Load again for optimization
    onnx_model = onnx.load_model(output_file)
    try:
        onnx_model, _ = onnxsim.simplify(onnx_model)
    except RuntimeError:
        pass
    if remove_init:
        # We did "keep_initializers_as_inputs=True" for optimizer
        # But let's now remove initializers from inputs
        onnx_model = remove_initializer_from_input(onnx_model)
    onnx.save_model(onnx_model, output_file)
    return onnx_model
