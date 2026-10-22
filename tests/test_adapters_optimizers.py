from dataclasses import replace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from gfg_instruments.adapters import TorchAdapter, ResNetAdapter
from gfg_instruments.adapters.resnet import CifarResNet18
from gfg_instruments.contracts import Batch, BoundaryContract, Component, ExecutionContract
from gfg_instruments import csrg, forks, optimizers
from gfg_instruments.errors import MeasurementError
from gfg_instruments.identity import digest


class OtherNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(4, 4) for _ in range(4))
        self.head = nn.Linear(4, 3)

    def forward(self, x, gates=()):
        for i, block in enumerate(self.blocks):
            x = x + (0 if str(i) in gates else torch.tanh(block(x)))
        return self.head(x)


class ExternalAdapter(TorchAdapter):
    """A third-party-style adapter defined outside the instrument package."""
    def __init__(self, optimizer_class, **kwargs):
        model = OtherNetwork()
        optimizer = optimizer_class(model.parameters(), lr=.01, **kwargs)
        components = tuple(Component(str(i), f"blocks.{i}:residual-output",
                            tuple(n for n, _ in model.named_parameters() if n.startswith(f"blocks.{i}."))) for i in range(4))
        super().__init__(model, optimizer, BoundaryContract(), components)

    def _forward(self, batch, gates):
        return self.model(batch.inputs[0], gates)

    def training_forward(self, batch):
        output = self.model(batch.inputs[0])
        return output, F.cross_entropy(output, batch.targets)


def small_batch():
    return Batch(tuple(f"other:{i}" for i in range(6)), (torch.randn(6, 4),), torch.arange(6) % 3,
                 tuple(f"group:{i % 3}" for i in range(6)), "unit-generated-vector-input")


@pytest.mark.parametrize("optimizer_class,kwargs", [
    (torch.optim.Adam, {"weight_decay": .1, "foreach": False}),
    (torch.optim.AdamW, {"weight_decay": .1, "foreach": False}),
    (torch.optim.SGD, {"momentum": .9, "dampening": 0.}),
    (torch.optim.SGD, {"momentum": .9, "dampening": .2}),
    (torch.optim.SGD, {"momentum": .9, "nesterov": True}),
])
def test_separate_optimizer_memory_against_native(optimizer_class, kwargs):
    adapter = ExternalAdapter(optimizer_class, **kwargs)
    batch = small_batch()
    adapter.native_update(batch)
    step = adapter.native_update(batch)
    transplanted = optimizers.transport(step.pre, step.pre, step.post, adapter.optimizer_semantics())
    for index, state in step.post.optimizer["state"].items():
        for key, expected in state.items():
            torch.testing.assert_close(transplanted.optimizer["state"][index][key], expected, rtol=2e-6, atol=1e-8)
    assert digest(transplanted.parameters) == digest(step.pre.parameters)
    assert digest(transplanted.buffers) == digest(step.pre.buffers)
    # Exact native execution with a restored complete state reproduces its result.
    adapter.restore(step.pre)
    replay = adapter.native_update(batch)
    assert replay.post.identity == step.post.identity


def test_uninitialized_memory_is_explicit_failure():
    adapter = ExternalAdapter(torch.optim.AdamW)
    batch = small_batch()
    step = adapter.native_update(batch)
    with pytest.raises(MeasurementError, match="UNINITIALIZED"):
        optimizers.transport(step.pre, step.pre, step.post, "adamw")


def test_optimizer_unknown_and_config_drift_rejected():
    with pytest.raises(MeasurementError, match="UNSUPPORTED_OPTIMIZER"):
        ExternalAdapter(torch.optim.RMSprop)
    with pytest.raises(MeasurementError, match="AMSGRAD"):
        ExternalAdapter(torch.optim.AdamW, amsgrad=True)
    adapter = ExternalAdapter(torch.optim.SGD, momentum=.9)
    batch = small_batch()
    adapter.native_update(batch)
    step = adapter.native_update(batch)
    bad = step.pre.clone()
    bad.optimizer["param_groups"][0]["momentum"] = .8
    with pytest.raises(MeasurementError, match="CONFIG"):
        optimizers.transport(bad, step.pre, step.post, "sgd_momentum")


def test_external_adapter_uses_unchanged_core(journal):
    adapter = ExternalAdapter(torch.optim.SGD, momentum=.9)
    batch = small_batch()
    result = csrg.run_csrg(adapter, batch, journal)
    assert csrg.validate_csrg(result)["status"] == "PASS"
    assert all(v["status"] == "UNDEFINED" for v in result["component_loads"].values())


def test_resnet_real_bn_and_parameter_only_path(journal):
    model = CifarResNet18(num_classes=5)
    adapter = ResNetAdapter(model, torch.optim.SGD(model.parameters(), lr=.01, momentum=.9))
    batch = Batch(("image:0", "image:1"), (torch.randn(2, 3, 16, 16),), torch.tensor([1, 2]),
                  ("label:1", "label:2"), "generated-image-smoke-not-cifar")
    entry = adapter.snapshot()
    step = adapter.native_update(batch, ExecutionContract(gradient_clip=None))
    assert any(torch.count_nonzero(v).item() for v in step.buffer_delta.values())
    path = forks.parameter_state(step.pre, step.parameter_delta)
    assert digest(path.buffers) == digest(step.pre.buffers)
    assert digest(path.buffers) != digest(step.post.buffers)
    assert all(torch.equal(path.parameters[n], step.pre.parameters[n] + step.parameter_delta[n]) for n in path.parameters)
    adapter.restore(entry)
    assert adapter.snapshot().identity == entry.identity
    result = csrg.run_csrg(adapter, batch, journal)
    assert csrg.validate_csrg(result)["status"] == "PASS"
    # Original stage gate function and adapted forward are the same real execution.
    model.eval()
    with torch.no_grad():
        for row in result["forwards"]:
            gates = tuple(0. if c.identity in row["gates"] else 1. for c in adapter.components)
            assert torch.equal(model(batch.inputs[0], gates), row["output"])


def test_residual_boundary_uses_frozen_candidates():
    from gfg_instruments.boundaries import readout
    truth = torch.randn(4, 3, 2, 2)
    candidates = truth[:, None] + .1 * torch.randn(4, 7, 3, 2, 2)
    batch = Batch(tuple(f"image:{i}/t:10/noise:{i}" for i in range(4)), (truth,), truth,
                  ("g0", "g0", "g1", "g1"), "identified-noise-unit", extras={"candidates": candidates})
    contract = BoundaryContract("ddpm-residual-candidate-v1", "residual_candidates", "nonnegative", True)
    result = readout(truth, truth, batch, contract)
    assert result["correct"].all()
    assert (result["margins"] > 0).all()
