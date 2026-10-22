from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from .errors import require
from .identity import digest


@dataclass(frozen=True)
class Component:
    identity: str
    intervention_site: str
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class BoundaryContract:
    """A frozen task readout. Ties are explicit; no inferred class ontology."""
    name: str = "classification-logit-margin-v1"
    kind: str = "classification"
    correct_rule: str = "argmax"
    zero_evaluable: bool = False
    group_quantile: float = 0.1

    def __post_init__(self):
        require(bool(self.name), "BOUNDARY_CONTRACT_ID_EMPTY")
        require(self.kind in ("classification", "residual_candidates"), "UNSUPPORTED_BOUNDARY_KIND")
        require(self.correct_rule in ("argmax", "positive", "nonnegative"), "UNSUPPORTED_CORRECTNESS_RULE")
        require(self.kind != "residual_candidates" or self.correct_rule != "argmax", "RESIDUAL_ARGMAX_UNDEFINED")
        require(self.group_quantile == 0.1, "CSRG_V1_QUANTILE_MUST_REMAIN_POINT_ONE")

    @property
    def identity(self):
        return digest(self)


@dataclass(frozen=True)
class Batch:
    identities: tuple[str, ...]
    inputs: tuple[torch.Tensor, ...]
    targets: torch.Tensor
    groups: tuple[str, ...]
    source: str
    extras: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.identities)
        require(n > 0 and len(set(self.identities)) == n, "EMPTY_OR_DUPLICATE_TARGET_IDENTITY")
        require(all(isinstance(key, str) and key for key in self.identities), "INVALID_TARGET_IDENTITY")
        require(bool(self.source), "SOURCE_AUTHORITY_REQUIRED")
        require(len(self.groups) == n and all(isinstance(g, str) and g for g in self.groups), "INVALID_GROUP_IDENTITIES")
        require(self.inputs and all(x.ndim > 0 and x.shape[0] == n for x in self.inputs), "INPUT_IDENTITY_COUNT_MISMATCH")
        require(self.targets.ndim > 0 and self.targets.shape[0] == n, "TARGET_IDENTITY_COUNT_MISMATCH")
        require(all(x.ndim > 0 and x.shape[0] == n for x in self.extras.values()), "EXTRA_IDENTITY_COUNT_MISMATCH")

    def item_hashes(self):
        return {key: digest({"source": self.source, "inputs": [v[i] for v in self.inputs],
                             "target": self.targets[i], "group": self.groups[i],
                             "extras": {k: v[i] for k, v in self.extras.items()}})
                for i, key in enumerate(self.identities)}

    @property
    def identity(self):
        return digest({"order": self.identities, "items": self.item_hashes()})

    def reordered(self, order):
        index = torch.as_tensor(order, dtype=torch.long)
        require(sorted(index.tolist()) == list(range(len(self.identities))), "NOT_A_BATCH_PERMUTATION")
        return Batch(tuple(self.identities[i] for i in order),
                     tuple(v[index.to(v.device)] for v in self.inputs),
                     self.targets[index.to(self.targets.device)], tuple(self.groups[i] for i in order),
                     self.source, {k: v[index.to(v.device)] for k, v in self.extras.items()})


@dataclass(frozen=True)
class ExecutionContract:
    name: str = "native-training-v1"
    gradient_clip: float | None = 1.0
    training_mode: str = "train"
    state_scope: str = "parameters_and_buffers"

    def __post_init__(self):
        require(self.training_mode in ("train", "eval"), "UNKNOWN_TRAINING_MODE")
        require(self.state_scope == "parameters_and_buffers", "FULL_NATIVE_STATE_REQUIRED")
        require(self.gradient_clip is None or self.gradient_clip > 0, "INVALID_GRADIENT_CLIP")


@runtime_checkable
class ModelAdapter(Protocol):
    components: tuple[Component, ...]
    boundary: BoundaryContract
    def snapshot(self) -> Any: ...
    def restore(self, snapshot: Any) -> None: ...
    def forward(self, batch: Batch, gates: tuple[str, ...] = ()) -> torch.Tensor: ...
    def native_update(self, batch: Batch, contract: ExecutionContract) -> Any: ...
    def evaluation_targets(self, batch: Batch) -> torch.Tensor: ...
    def optimizer_semantics(self) -> Any: ...
