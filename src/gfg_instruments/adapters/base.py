from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from ..contracts import ExecutionContract
from ..errors import require
from .. import optimizers
from ..state import Snapshot, clone, snapshot, restore


@dataclass
class NativeUpdate:
    pre: Snapshot
    post: Snapshot
    parameter_delta: dict
    buffer_delta: dict
    raw_gradients: dict
    gradients: dict
    training_output: torch.Tensor
    loss: float
    batch_identity: str
    contract: ExecutionContract


class TorchAdapter(ABC):
    def __init__(self, model, optimizer, boundary, components):
        self.model, self.optimizer, self.boundary = model, optimizer, boundary
        self.components = tuple(components)
        require(len(self.components) == 4 and len({c.identity for c in self.components}) == 4, "CSRG_REQUIRES_FOUR_DISTINCT_COMPONENTS")
        names = dict(model.named_parameters())
        require(all(c.intervention_site and c.parameter_names and set(c.parameter_names) <= set(names)
                    for c in self.components), "INVALID_COMPONENT_REGISTRY")
        self._optimizer_kind = optimizers.kind(optimizer)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def optimizer_semantics(self):
        return self._optimizer_kind

    def snapshot(self):
        return snapshot(self.model, self.optimizer)

    def restore(self, value):
        restore(self.model, self.optimizer, value)
        require(self.snapshot().identity == value.identity, "RESTORE_NOT_EXACT")

    @abstractmethod
    def _forward(self, batch, gates): ...

    @abstractmethod
    def training_forward(self, batch):
        """Return native model output and scalar loss, without calling backward."""

    def evaluation_targets(self, batch):
        return batch.targets

    @torch.no_grad()
    def forward(self, batch, gates=()):
        require(len(set(gates)) == len(gates) and set(gates) <= {c.identity for c in self.components}, "UNKNOWN_OR_DUPLICATE_GATE")
        modes = {n: m.training for n, m in self.model.named_modules()}
        self.model.eval()
        try:
            return self._forward(batch, gates).detach().cpu().clone()
        finally:
            for name, module in self.model.named_modules():
                module.training = modes[name]

    def native_update(self, batch, contract=ExecutionContract()):
        pre = self.snapshot()
        try:
            self.model.train(contract.training_mode == "train")
            self.optimizer.zero_grad(set_to_none=True)
            output, loss = self.training_forward(batch)
            require(loss.ndim == 0 and torch.isfinite(loss), "INVALID_NATIVE_LOSS")
            loss.backward()
            raw = {n: clone(v.grad) for n, v in self.model.named_parameters()}
            require(any(v is not None for v in raw.values()), "NO_NATIVE_GRADIENT")
            require(all(v is None or torch.isfinite(v).all() for v in raw.values()), "NONFINITE_NATIVE_GRADIENT")
            if contract.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), contract.gradient_clip)
            gradients = {n: clone(v.grad) for n, v in self.model.named_parameters()}
            self.optimizer.step()  # the actual optimizer, exactly once
            post = self.snapshot()
            require(all(torch.isfinite(v).all() for v in post.parameters.values()), "NONFINITE_NATIVE_UPDATE")
            return NativeUpdate(pre, post,
                                {n: post.parameters[n] - v for n, v in pre.parameters.items()},
                                {n: post.buffers[n] - v for n, v in pre.buffers.items()}, raw, gradients,
                                output.detach().cpu().clone(), float(loss.detach()), batch.identity, contract)
        except Exception:
            self.restore(pre)
            raise
