from __future__ import annotations

import torch.nn.functional as F

from ..contracts import BoundaryContract, Component
from .._vendor.resnet_model import CifarResNet18
from .base import TorchAdapter


class ResNetAdapter(TorchAdapter):
    """Frozen research implementation: zero residual contributions, not whole stages."""
    def __init__(self, model, optimizer, boundary=BoundaryContract()):
        components = [Component(path, path + ":each-block-residual-before-addition",
                                tuple(n for n, _ in model.named_parameters() if n.startswith(path + ".")))
                      for path in model.component_names]
        super().__init__(model, optimizer, boundary, components)

    def _forward(self, batch, gates):
        mask = tuple(0.0 if c.identity in gates else 1.0 for c in self.components)
        return self.model(batch.inputs[0].to(self.device), mask)

    def training_forward(self, batch):
        logits = self.model(batch.inputs[0].to(self.device))
        return logits, F.cross_entropy(logits, batch.targets.to(self.device))


__all__ = ["ResNetAdapter", "CifarResNet18"]
