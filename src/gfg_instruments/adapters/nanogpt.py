from __future__ import annotations

import torch

from ..contracts import BoundaryContract, Component
from ..errors import require
from .._vendor.nanogpt.model import GPT, GPTConfig
from .base import TorchAdapter


class NanoGPTAdapter(TorchAdapter):
    """Unmodified frozen nanoGPT; gate component output before residual addition."""
    def __init__(self, model, optimizer, boundary=BoundaryContract(), component_paths=None):
        paths = component_paths or ("transformer.h.0.attn", "transformer.h.0.mlp", "transformer.h.1.attn", "transformer.h.1.mlp")
        self.modules = {path: model.get_submodule(path) for path in paths}
        components = [Component(path, path + ":output-before-residual-add",
                                tuple(n for n, _ in model.named_parameters() if n.startswith(path + "."))) for path in paths]
        super().__init__(model, optimizer, boundary, components)

    def _forward(self, batch, gates):
        hooks = [self.modules[name].register_forward_hook(lambda _m, _i, out: torch.zeros_like(out)) for name in gates]
        try:
            # Passing full labels preserves the historical full-sequence forward.
            logits, _ = self.model(batch.inputs[0].to(self.device), batch.targets.to(self.device))
            return logits[:, -1, :]
        finally:
            for hook in hooks:
                hook.remove()

    def training_forward(self, batch):
        require(batch.targets.shape == batch.inputs[0].shape, "NANOGPT_SEQUENCE_TARGETS_REQUIRED")
        return self.model(batch.inputs[0].to(self.device), batch.targets.to(self.device))

    def evaluation_targets(self, batch):
        return batch.targets[:, -1]


__all__ = ["NanoGPTAdapter", "GPT", "GPTConfig"]
