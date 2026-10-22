from copy import deepcopy
import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from gfg_instruments import csrg, forks, ledger, paths
from gfg_instruments.contracts import Batch, BoundaryContract, ExecutionContract
from gfg_instruments.errors import MeasurementError
from gfg_instruments.evidence import Journal, payload, validate_journal
from gfg_instruments.identity import digest
from gfg_instruments.state import assert_isolated

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "provenance/upstream/experiments"


def original_function(relative, name, globals):
    source = (UPSTREAM / relative).read_text("utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    env = dict(globals)
    exec(compile(ast.fix_missing_locations(code), str(relative), "exec"), env)
    return env[name]


def test_csrg_real_execution_and_restoration(adapter, batch, journal):
    before = adapter.snapshot().identity
    result = csrg.run_csrg(adapter, batch, journal)
    assert csrg.validate_csrg(result)["status"] == "PASS"
    assert result["actual_forward_count"] == 12
    assert result["metrics"]["necessity"].shape == (4, 7)
    assert adapter.snapshot().identity == before
    again = csrg.run_csrg(adapter, batch, journal)
    np.testing.assert_array_equal(result["metrics"]["necessity"], again["metrics"]["necessity"])
    assert len({r["id"] for r in journal.events}) == len(journal.events)


def test_frozen_212_by_24_readout_exact_parity():
    original = original_function("gfg_nanogpt_support_redundancy_v1/builder.py", "decision_outputs",
                                 {"np": np, "torch": torch, "require": lambda c, m: c or (_ for _ in ()).throw(AssertionError(m))})
    logits = torch.randn(212, 24)
    targets = torch.full((212, 3), -1, dtype=torch.long)
    targets[:, -1] = torch.arange(212) % 23
    batch = Batch(tuple(f"legacy:{i}" for i in range(212)), (torch.zeros(212, 3),), targets,
                  tuple(f"group:{int(t):02d}" for t in targets[:, -1]), "independent-test-not-historical")
    from gfg_instruments.boundaries import readout
    new = readout(logits, targets[:, -1], batch, BoundaryContract())
    old = original(logits, targets)
    for key in ("margins", "predictions", "group_q10_margin"):
        np.testing.assert_array_equal(new[key], old[key])


def test_support_kernel_frozen_code_exact_parity():
    # Execute the exact original arithmetic block, not a second hand-written formula.
    source = (UPSTREAM / "gfg_nanogpt_stepwise_support_transition_v1/runtime.py").read_text("utf-8")
    start = source.index("            necessity = np.stack", source.index("    def support_probe"))
    end = source.index("            snapshot = self.snapshot()", start)
    import textwrap
    block = textwrap.dedent(source[start:end])
    ids = ("a", "b", "c", "d")
    from itertools import combinations
    singles, pairs = [(i,) for i in ids], list(combinations(ids, 2))
    rng = np.random.default_rng(123)
    baseline = rng.normal(size=23)
    gates = {g: {"group_q10_margin": rng.normal(size=23)} for g in singles + pairs}
    from types import SimpleNamespace
    env = dict(np=np, baseline_group=baseline, singles=singles, pairs=pairs, by_gate=gates,
               self=SimpleNamespace(probe_contract=SimpleNamespace(target_group_count=23)))
    exec(block, env)
    new = csrg.support_metrics(baseline, {g: gates[g]["group_q10_margin"] for g in singles},
                               {g: gates[g]["group_q10_margin"] for g in pairs}, ids, 23)
    for a, b in (("necessity", "necessity"), ("backup", "backup"), ("single_failure_slack", "single_slack"),
                 ("double_failure_slack", "double_slack"), ("effective_support", "effective")):
        np.testing.assert_array_equal(new[a], env[b])
    for key in ("allocation", "concentration"):
        np.testing.assert_array_equal(new[key][..., new["defined"]], env[key][..., new["defined"]])


def test_undefined_support_not_zero():
    from itertools import combinations
    ids = ("a", "b", "c", "d")
    values = np.ones(2)
    result = csrg.support_metrics(values, {(i,): values for i in ids}, {p: values for p in combinations(ids, 2)}, ids, 2)
    assert not result["defined"].any()
    assert np.isnan(result["effective_support"]).all() and np.isnan(result["concentration"]).all()
    assert paths.finite_rms(np.array([np.nan, np.inf])) is None


def test_two_gates_not_shapley_and_corruption_rejected(adapter, batch, journal):
    result = csrg.run_csrg(adapter, batch, journal)
    result["metrics"]["necessity"][0, 0] += .01
    with pytest.raises(MeasurementError, match="NECESSITY"):
        csrg.validate_csrg(result)


def test_forks_real_native_update_and_horizon(adapter, batch, journal):
    donor = forks.capture_update(adapter, batch, journal)
    before = adapter.snapshot()
    result = forks.run_forks(adapter, batch, batch, journal, donor=donor, horizons=(1, 2), continuation_batches=(batch,))
    assert result["branches"] == forks.BRANCHES
    assert adapter.snapshot().identity == before.identity
    assert result["observations"]["1"]["skip"]["state_identity"] == before.identity
    for probes in result["observations"].values():
        for probe in probes.values():
            csrg.validate_csrg(probe)
    assert result["branch_storage_isolated"]


def test_amplitude_real_and_legacy_kernel_ast(adapter, batch, journal):
    original = ast.parse((UPSTREAM / "gfg_nanogpt_stepwise_support_transition_v1/amplitude_path.py").read_text("utf-8"))
    new = ast.parse((ROOT / "src/gfg_instruments/amplitude_kernel.py").read_text("utf-8"))
    get = lambda tree: next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_numeric_path")
    assert ast.dump(get(original)) == ast.dump(get(new))
    update = forks.capture_update(adapter, batch, journal)
    adapter.restore(update[0].pre)
    before = adapter.snapshot().identity
    result = paths.run_amplitude_path(adapter, batch, journal, realized_update=update, horizons=(1, 2), continuation_batches=(batch,))
    assert paths.validate_path(result)["status"] == "PASS"
    assert adapter.snapshot().identity == before
    assert len(result["probes"]) == 11 and len(result["continuations"]) == 5


def test_identity_ledger_real_pretarget_order(adapter, batch, journal):
    update = forks.capture_update(adapter, batch, journal)
    adapter.restore(update[0].pre)
    before = adapter.snapshot().identity
    result = ledger.run_boundary_ledger(adapter, batch, journal, realized_update=update)
    assert result["adjudication"]["evaluable_count"] == 8
    operations = [r["operation"] for r in journal.events]
    assert operations.index("pretarget_prediction_seal") < operations.index("postseal_parameter_endpoint")
    assert adapter.snapshot().identity == before


def test_ledger_reorders_by_identity_not_position(adapter, batch, journal):
    read, ref = csrg.measure(adapter, batch, journal)
    seal = ledger.seal_prediction(batch, adapter.boundary, read, read["margins"], journal, [ref], method="identity-test")
    shuffled = batch.reordered(list(reversed(range(8))))
    actual, actual_ref = csrg.measure(adapter, shuffled, journal)
    result = ledger.adjudicate(seal, actual, shuffled, adapter.boundary, journal, actual_ref)
    assert result["accuracy"] == 1.
    wrong = Batch(shuffled.identities, shuffled.inputs, shuffled.targets + 1, shuffled.groups, shuffled.source)
    with pytest.raises(MeasurementError, match="CONTENT"):
        ledger.adjudicate(seal, actual, wrong, adapter.boundary, journal, actual_ref)
    seal["predicted_margins"][0] += 1
    with pytest.raises(MeasurementError, match="SEAL_CHANGED"):
        ledger.adjudicate(seal, actual, shuffled, adapter.boundary, journal, actual_ref)


def test_zero_prediction_is_not_false(adapter, batch, journal):
    read, ref = csrg.measure(adapter, batch, journal)
    seal = ledger.seal_prediction(batch, adapter.boundary, read, np.zeros(8), journal, [ref], method="zero-negative-control")
    actual, ar = csrg.measure(adapter, batch, journal)
    result = ledger.adjudicate(seal, actual, batch, adapter.boundary, journal, ar)
    assert result["accuracy"] is None and result["not_evaluable_count"] == 8
    assert all(r["predicted"] is None for r in result["rows"])


def test_none_reciprocal_cannot_mean_receiving_dominant():
    from itertools import product
    rows = [dict(recipient=r, transferred_branch=b, horizon=h, aggregate_donor_closer=None)
            for r, b, h in product(("A", "B"), ("donor_parameter_delta", "donor_optimizer_innovation", "donor_joint_update"), (1, 2, 5, 20, 100))]
    original = original_function("gfg_nanogpt_stepwise_support_transition_v1/reciprocal_validator.py", "_classification", {})
    assert original(rows)["decision"] == "RECEIVING_STATE_DOMINANT"
    assert forks.classify_reciprocal(rows)["decision"] == "NOT_EVALUABLE"
    for value in (True, False):
        for row in rows:
            row["aggregate_donor_closer"] = value
        assert forks.classify_reciprocal(rows)["decision"] == original(rows)["decision"]


def test_state_alias_invalid_gate_and_nonfinite_rejected(adapter, batch):
    before = adapter.snapshot()
    with pytest.raises(MeasurementError, match="ALIAS"):
        assert_isolated(before, before)
    assert_isolated(before, before.clone())
    with pytest.raises(MeasurementError, match="GATE"):
        adapter.forward(batch, ("not-a-component",))
    bad = before.clone()
    next(iter(bad.parameters.values())).flatten()[0] = float("nan")
    with pytest.raises(MeasurementError, match="NONFINITE"):
        adapter.restore(bad)
    assert adapter.snapshot().identity == before.identity


def test_duplicate_sample_identity_rejected(batch):
    with pytest.raises(MeasurementError, match="DUPLICATE"):
        Batch(("same",) * 8, batch.inputs, batch.targets, batch.groups, batch.source)


def test_journal_hash_and_state_payload_preserved(adapter, batch, journal):
    update, _ = forks.capture_update(adapter, batch, journal)
    csrg.run_csrg(adapter, batch, journal)
    journal.save()
    validate_journal(journal.root)
    first = journal.events[0]
    record = payload(journal.root, first["artifact"])
    assert digest(record["parameter_delta"]) == digest(update.parameter_delta)
    path = journal.root / first["artifact"]["path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(MeasurementError, match="HASH"):
        validate_journal(journal.root)


def test_snapshot_journal_roundtrip_and_invalid_memory(adapter, batch, journal):
    from gfg_instruments.state import Snapshot
    state = adapter.snapshot()
    ref = journal.artifact(state)
    reloaded = Snapshot.from_record(payload(journal.root, ref))
    assert reloaded.identity == state.identity
    adapter.native_update(batch, ExecutionContract())
    adapter.restore(reloaded)
    assert adapter.snapshot().identity == state.identity
    bad = state.clone()
    next(iter(bad.optimizer["state"].values()))["exp_avg"].flatten()[0] = float("nan")
    with pytest.raises(MeasurementError, match="OPTIMIZER_STATE"):
        adapter.restore(bad)
    assert adapter.snapshot().identity == state.identity


def test_recorded_margin_tampering_rejected(adapter, batch, journal):
    result = csrg.run_csrg(adapter, batch, journal)
    result["forwards"][0]["margins"][0] += 1.
    with pytest.raises(MeasurementError, match="READOUT_RECOMPUTATION"):
        csrg.validate_csrg(result)


def test_native_update_provenance_cannot_be_substituted(adapter, batch, journal):
    update, ref = forks.capture_update(adapter, batch, journal)
    next(iter(update.parameter_delta.values())).flatten()[0] += 1.
    with pytest.raises(MeasurementError, match="RECEIPT_MISMATCH"):
        forks.verify_realized_update(update, ref, journal)


def test_target_read_before_seal_rejected(adapter, batch, journal):
    read, ref = csrg.measure(adapter, batch, journal)
    seal = ledger.seal_prediction(batch, adapter.boundary, read, read["margins"], journal, [ref], method="negative-control")
    with pytest.raises(MeasurementError, match="FOLLOW_SEAL"):
        ledger.adjudicate(seal, read, batch, adapter.boundary, journal, ref)


def test_full_frozen_path_role_coverage(adapter, batch, journal):
    probe = csrg.run_csrg(adapter, batch, journal)
    arrays = forks.response_arrays(probe)
    assert len(arrays) == 12 * 5 + 7
    for index in range(12):
        assert f"forward/{index}/logits" in arrays and f"forward/{index}/group_membership" in arrays


def test_reciprocal_signature_frozen_function_parity():
    from statistics import median
    t, r, d = ({"a": np.array([1., 4., np.nan])}, {"a": np.array([2., 3., np.nan])}, {"a": np.array([-1., 5., np.nan])})
    def effect(root, label, horizon, branch):
        return t if branch == "donor_parameter_delta" else (r if label == "A" else d)
    original = original_function("gfg_nanogpt_stepwise_support_transition_v1/reciprocal_validator.py", "_signature_row",
                                 {"np": np, "median": median, "_effect": effect, "require": lambda c, m: c or (_ for _ in ()).throw(AssertionError(m)),
                                  "_rms": lambda value: float(np.sqrt(np.mean(np.square(value, dtype=np.float64), dtype=np.float64)))})
    options = dict(recipient="A", donor="B", horizon=1, transferred_branch="donor_parameter_delta", native_branch="native_parameter_only")
    assert forks.signature_row(t, r, d, **options) == original(None, **options)
    assert forks.signature_row(r, r, r, **options)["aggregate_donor_closer"] is None
