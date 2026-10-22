from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random

import numpy as np
import torch

from .errors import require
from .identity import digest


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone(v) for v in value)
    return deepcopy(value)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone(),
                cuda=[v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if value["cuda"]:
        require(torch.cuda.is_available() and len(value["cuda"]) == torch.cuda.device_count(), "CUDA_RNG_LAYOUT_MISMATCH")
        torch.cuda.set_rng_state_all(value["cuda"])


@dataclass
class Snapshot:
    parameters: dict
    buffers: dict
    optimizer: dict
    optimizer_names: tuple[tuple[str, ...], ...]
    optimizer_kind: str
    modes: dict
    rng: dict
    gradients: dict

    def clone(self):
        return deepcopy(self)

    @classmethod
    def from_record(cls, value):
        """Rehydrate the lossless JSON/tensor journal, without pickle execution."""
        material = clone(value)
        material["optimizer_names"] = tuple(tuple(group) for group in material["optimizer_names"])
        material["optimizer"]["state"] = {int(k): v for k, v in material["optimizer"]["state"].items()}
        def tuples(v):
            return tuple(tuples(c) for c in v) if isinstance(v, list) else v
        material["rng"]["python"] = tuples(material["rng"]["python"])
        material["rng"]["numpy"] = tuple(material["rng"]["numpy"])
        return cls(**material)

    @property
    def identity(self):
        return digest(self.__dict__)

    @property
    def model_identity(self):
        return digest({"parameters": self.parameters, "buffers": self.buffers})


def optimizer_names(model, optimizer):
    names = {id(v): name for name, v in model.named_parameters()}
    require(all(id(v) in names for g in optimizer.param_groups for v in g["params"]), "OPTIMIZER_CONTAINS_FOREIGN_PARAMETER")
    result = tuple(tuple(names[id(v)] for v in g["params"]) for g in optimizer.param_groups)
    flat = [n for g in result for n in g]
    require(len(flat) == len(set(flat)), "DUPLICATE_OPTIMIZER_PARAMETER")
    return result


def snapshot(model, optimizer):
    return Snapshot(clone(dict(model.named_parameters())), clone(dict(model.named_buffers())),
                    clone(optimizer.state_dict()), optimizer_names(model, optimizer),
                    type(optimizer).__module__ + "." + type(optimizer).__qualname__,
                    {n: m.training for n, m in model.named_modules()}, rng_state(),
                    {n: clone(v.grad) for n, v in model.named_parameters()})


def restore(model, optimizer, value: Snapshot):
    require(value.optimizer_kind == type(optimizer).__module__ + "." + type(optimizer).__qualname__, "OPTIMIZER_KIND_MISMATCH")
    require(value.optimizer_names == optimizer_names(model, optimizer), "OPTIMIZER_PARAMETER_IDENTITY_MISMATCH")
    # Validate the entire layout before copying any tensor.
    for actual, saved in ((dict(model.named_parameters()), value.parameters),
                          (dict(model.named_buffers()), value.buffers)):
        require(set(actual) == set(saved), "STATE_NAME_SET_MISMATCH")
        for name, tensor in actual.items():
            require(tensor.shape == saved[name].shape and tensor.dtype == saved[name].dtype,
                    f"STATE_LAYOUT_MISMATCH:{name}")
            if saved[name].is_floating_point():
                require(torch.isfinite(saved[name]).all(), f"NONFINITE_STATE:{name}")
    require(set(value.modes) == set(dict(model.named_modules())), "MODULE_MODE_LAYOUT_MISMATCH")
    require(set(value.gradients) == set(value.parameters), "GRADIENT_NAME_SET_MISMATCH")
    for name, gradient in value.gradients.items():
        require(gradient is None or (gradient.shape == value.parameters[name].shape
                and gradient.dtype == value.parameters[name].dtype and torch.isfinite(gradient).all()), "INVALID_SAVED_GRADIENT")
    require(len(value.optimizer["param_groups"]) == len(value.optimizer_names), "SAVED_OPTIMIZER_GROUP_MISMATCH")
    owned = {}
    for names, group in zip(value.optimizer_names, value.optimizer["param_groups"]):
        require(len(names) == len(group["params"]), "SAVED_OPTIMIZER_GROUP_MISMATCH")
        owned.update(zip(group["params"], names))
    require(set(value.optimizer["state"]) <= set(owned), "SAVED_OPTIMIZER_FOREIGN_STATE")
    for key, memory in value.optimizer["state"].items():
        for field, tensor in memory.items():
            require(isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all(), "NONFINITE_OR_UNSUPPORTED_OPTIMIZER_STATE")
            require(tensor.numel() == 1 if field == "step" else tensor.shape == value.parameters[owned[key]].shape,
                    "SAVED_OPTIMIZER_TENSOR_LAYOUT")
    # Validate RNG structure without changing the process-wide generators.
    random.Random().setstate(value.rng["python"])
    np.random.RandomState().set_state(value.rng["numpy"])
    torch.Generator().set_state(value.rng["torch"])
    require(not value.rng["cuda"] or (torch.cuda.is_available()
            and len(value.rng["cuda"]) == torch.cuda.device_count()), "CUDA_RNG_LAYOUT_MISMATCH")
    with torch.no_grad():
        for actual, saved in ((dict(model.named_parameters()), value.parameters),
                              (dict(model.named_buffers()), value.buffers)):
            for name, tensor in actual.items():
                tensor.copy_(saved[name].to(tensor.device))
    optimizer.load_state_dict(clone(value.optimizer))
    for name, module in model.named_modules():
        module.training = value.modes[name]
    for name, parameter in model.named_parameters():
        gradient = value.gradients[name]
        parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
    restore_rng(value.rng)


def assert_isolated(*snapshots):
    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from tensors(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                yield from tensors(child)
    seen = set()
    for state in snapshots:
        pointers = {t.untyped_storage().data_ptr() for t in tensors(state.__dict__) if t.numel()}
        require(not seen.intersection(pointers), "BRANCH_STATE_STORAGE_ALIAS")
        seen.update(pointers)
