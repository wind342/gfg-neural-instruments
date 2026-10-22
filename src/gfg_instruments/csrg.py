"""CSRG-4C: frozen q10/single/pair formulas with model bindings removed."""
from __future__ import annotations

from itertools import combinations
import numpy as np
import torch

from .boundaries import readout, validate_readout
from .errors import require
from .identity import digest
from .optimizers import component_loads


def measure(adapter, batch, journal, gates=(), state_ref=None):
    before = adapter.snapshot()
    state_ref = state_ref or journal.source("receiving-state:" + before.identity, before)
    batch_ref = journal.source("evaluation-batch:" + batch.identity, batch)
    contract_ref = journal.source("boundary-contract:" + adapter.boundary.identity, adapter.boundary)
    output = adapter.forward(batch, gates)
    require(adapter.snapshot().identity == before.identity, "PROBE_MUTATED_STATE_OR_RNG")
    result = readout(output, adapter.evaluation_targets(batch), batch, adapter.boundary)
    result.update(gates=tuple(gates), state_identity=before.identity)
    ref = journal.record("native_component_gate_forward", [state_ref, batch_ref, contract_ref], result)
    return result, ref


def support_metrics(baseline_group, single_margins, pair_margins, component_ids, group_count):
    """Extracted from StepwiseTrainingRuntime.support_probe; same clipping and q10."""
    singles = tuple((value,) for value in component_ids)
    pairs = tuple(combinations(component_ids, 2))
    by_gate = {**{key: {"group_q10_margin": value} for key, value in single_margins.items()},
               **{key: {"group_q10_margin": value} for key, value in pair_margins.items()}}
    necessity = np.stack([np.maximum(0.0, baseline_group - by_gate[gate]["group_q10_margin"]) for gate in singles])
    single_index = {gate[0]: index for index, gate in enumerate(singles)}
    backup = np.stack([
        np.maximum(0.0, baseline_group - by_gate[gate]["group_q10_margin"]
                   - necessity[single_index[gate[0]]] - necessity[single_index[gate[1]]])
        for gate in pairs
    ])
    single_slack = np.min(np.stack([by_gate[gate]["group_q10_margin"] for gate in singles]), axis=0)
    double_slack = np.min(np.stack([by_gate[gate]["group_q10_margin"] for gate in pairs]), axis=0)
    total = necessity.sum(axis=0)
    defined = total > 0.0
    allocation = np.zeros_like(necessity, dtype=np.float64)
    allocation[:, defined] = necessity[:, defined] / total[defined]
    concentration = np.sum(allocation * allocation, axis=0)
    effective = np.full(group_count, np.nan, dtype=np.float64)
    effective[defined] = 1.0 / concentration[defined]
    # New explicit undefined mask: no scientific assertion of zero concentration
    # or zero allocation when there is no positive measured necessity.
    allocation[:, ~defined] = np.nan
    concentration[~defined] = np.nan
    return dict(necessity=necessity, backup=backup, single_failure_slack=single_slack,
                double_failure_slack=double_slack, allocation=allocation,
                concentration=concentration, effective_support=effective, defined=defined)


def run_csrg(adapter, batch, journal, *, state_ref=None):
    entry = adapter.snapshot()
    ids = tuple(c.identity for c in adapter.components)
    require(len(ids) == len(set(ids)) == 4, "CSRG_REQUIRES_FOUR_COMPONENTS")
    registry = journal.source("components:" + digest(adapter.components), adapter.components)
    forwards, refs = [], []
    plans = [(), ()] + [(c,) for c in ids] + list(combinations(ids, 2))
    try:
        for gate in plans:
            row, ref = measure(adapter, batch, journal, gate, state_ref)
            forwards.append(row)
            refs.append(ref)
        require(torch.equal(forwards[0]["output"], forwards[1]["output"]), "BASELINE_REPETITION_NOT_BYTE_EXACT")
        singles = {tuple(r["gates"]): r["group_q10_margin"] for r in forwards[2:6]}
        pairs = {tuple(r["gates"]): r["group_q10_margin"] for r in forwards[6:]}
        metrics = support_metrics(forwards[0]["group_q10_margin"], singles, pairs, ids, len(forwards[0]["group_order"]))
        result = dict(schema="csrg-4c-adapted-v1.1", component_ids=ids,
                      intervention_sites=[c.intervention_site for c in adapter.components],
                      boundary_id=adapter.boundary.identity, item_hashes=batch.item_hashes(),
                      group_order=forwards[0]["group_order"], actual_forward_count=12,
                      baseline_byte_exact=True, forwards=forwards, metrics=metrics,
                      capability_accuracy=float(np.mean(forwards[0]["correct"])),
                      state_identity=entry.identity, state_preserved=True,
                      optimizer_kind=adapter.optimizer_semantics(),
                      component_loads=component_loads(entry, adapter.components, adapter.optimizer_semantics()))
        ref = journal.record("csrg4c_scientific_derivation", [registry, *refs], result)
        for i, defined in enumerate(metrics["defined"]):
            if not defined:
                journal.record("undefined_effective_support", [ref], {"group": result["group_order"][i]},
                               disposition="ZERO_TOTAL_NECESSITY")
        result["evidence_ref"] = ref
        return result
    finally:
        adapter.restore(entry)


def validate_csrg(result):
    """Independent scalar/group recomputation, not a call back into support_metrics."""
    rows, metrics = result["forwards"], result["metrics"]
    require(len(rows) == 12 and result["actual_forward_count"] == 12, "CSRG_FORWARD_COVERAGE")
    require(torch.equal(rows[0]["output"], rows[1]["output"]), "CSRG_BASELINE_MISMATCH")
    for row in rows:
        validate_readout(row)
        require(row["state_identity"] == result["state_identity"] and row["boundary_id"] == result["boundary_id"]
                and row["item_hashes"] == result["item_hashes"], "CSRG_PROBE_IDENTITY_MISMATCH")
    ids = tuple(result["component_ids"])
    require(tuple(tuple(row["gates"]) for row in rows) == ((), (), *((c,) for c in ids), *combinations(ids, 2)), "CSRG_GATE_PLAN_MISMATCH")
    for group in range(len(result["group_order"])):
        m = rows[0]["group_q10_margin"][group]
        singles = [row["group_q10_margin"][group] for row in rows[2:6]]
        pairs = [row["group_q10_margin"][group] for row in rows[6:]]
        necessity = np.asarray([max(0., m - v) for v in singles])
        backup = [max(0., m - v - necessity[i] - necessity[j])
                  for (i, j), v in zip(combinations(range(4), 2), pairs)]
        require(np.array_equal(metrics["necessity"][:, group], necessity), "NECESSITY_RECOMPUTATION_FAILED")
        require(np.array_equal(metrics["backup"][:, group], backup), "BACKUP_RECOMPUTATION_FAILED")
        require(metrics["single_failure_slack"][group] == min(singles) and metrics["double_failure_slack"][group] == min(pairs), "SLACK_RECOMPUTATION_FAILED")
        if necessity.sum() == 0:
            require(not metrics["defined"][group] and np.isnan(metrics["effective_support"][group])
                    and np.isnan(metrics["concentration"][group]) and np.isnan(metrics["allocation"][:, group]).all(), "UNDEFINED_SUPPORT_WRONGLY_NUMERIC")
        else:
            weights = necessity / necessity.sum()
            require(metrics["defined"][group] and np.array_equal(weights, metrics["allocation"][:, group]), "ALLOCATION_RECOMPUTATION_FAILED")
            require(metrics["effective_support"][group] == 1. / np.sum(weights ** 2), "EFFECTIVE_SUPPORT_RECOMPUTATION_FAILED")
            require(metrics["concentration"][group] == np.sum(weights ** 2), "CONCENTRATION_RECOMPUTATION_FAILED")
    return {"status": "PASS", "checked_groups": len(result["group_order"]), "actual_forwards": 12}
