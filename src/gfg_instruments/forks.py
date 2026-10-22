from __future__ import annotations

import random
import numpy as np
import torch

from .contracts import ExecutionContract
from .csrg import run_csrg
from .errors import require
from .identity import digest
from .optimizers import transport
from .state import assert_isolated


BRANCHES = ("skip", "native_full", "native_parameter_only", "native_optimizer_only",
            "donor_parameter_delta", "donor_optimizer_innovation", "donor_joint_update")


def capture_update(adapter, batch, journal, contract=ExecutionContract(), *, state_ref=None):
    pre = adapter.snapshot()
    state_ref = state_ref or journal.source("receiving-state:" + pre.identity, pre)
    batch_ref = journal.source("training-batch:" + batch.identity, batch)
    config_ref = journal.source("execution-contract:" + digest(contract), contract)
    update = adapter.native_update(batch, contract)
    ref = journal.record("actual_native_optimizer_step", [state_ref, batch_ref, config_ref], update)
    return update, ref


def parameter_state(receiver, delta, scale=1.):
    result = receiver.clone()
    require(set(delta) == set(result.parameters), "UPDATE_PARAMETER_IDENTITY_MISMATCH")
    for name, value in result.parameters.items():
        require(delta[name].shape == value.shape and delta[name].dtype == value.dtype,
                "UPDATE_PARAMETER_LAYOUT_MISMATCH")
        require(torch.isfinite(delta[name]).all(), "NONFINITE_REALIZED_UPDATE")
        value.add_((delta[name] * scale).to(value.dtype))
    return result


def verify_realized_update(update, ref, journal):
    row = next((row for row in journal.events if row["id"] == ref), None)
    require(row is not None and row["operation"] == "actual_native_optimizer_step", "NOT_A_NATIVE_UPDATE_RECEIPT")
    require(journal.artifact(update) == row["artifact"], "REALIZED_UPDATE_RECEIPT_MISMATCH")
    require(set(update.parameter_delta) == set(update.pre.parameters) == set(update.post.parameters), "NATIVE_UPDATE_LAYOUT_MISMATCH")
    for name, delta in update.parameter_delta.items():
        require(torch.equal(delta, update.post.parameters[name] - update.pre.parameters[name]), "NATIVE_UPDATE_NOT_ACTUAL_POST_MINUS_PRE")


def response_arrays(probe):
    # Preserve the full frozen branch/path output set, not just baseline logits.
    result = {}
    group_ids = {g: i for i, g in enumerate(probe["group_order"])}
    for index, row in enumerate(probe["forwards"]):
        for key in ("margins", "predictions", "group_q10_margin"):
            result[f"forward/{index}/{key}"] = row[key]
        result[f"forward/{index}/logits"] = row["output"].numpy()
        result[f"forward/{index}/group_membership"] = np.asarray([group_ids[g] for g in row["group_membership"]], dtype=np.int64)
    mapping = {"necessity": "necessity", "pair_backup": "backup", "single_failure_slack": "single_failure_slack",
               "double_failure_slack": "double_failure_slack", "support_allocation": "allocation",
               "support_concentration": "concentration", "effective_support": "effective_support"}
    result.update({name: probe["metrics"][field] for name, field in mapping.items()})
    return result


def set_opportunity(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_forks(adapter, training_batch, evaluation_batch, journal, *, donor,
              contract=ExecutionContract(), horizons=(1,), continuation_batches=(), seed=2021):
    """Native full state is explicit; parameter/optimizer-only forks hold buffers.

    Donor states must belong to the same parameter identity layout and validated
    optimizer family. A donor is an actual captured NativeUpdate, never a fitted
    or nominal optimizer formula. Each receiver's future batches are identical
    across its branches, and RNG opportunities are matched explicitly.
    """
    require(horizons and tuple(sorted(set(horizons))) == tuple(horizons) and horizons[0] == 1,
            "INVALID_FORK_HORIZONS")
    require(len(continuation_batches) == max(horizons) - 1, "CONTINUATION_BATCH_COVERAGE")
    entry = adapter.snapshot()
    try:
        native, native_ref = capture_update(adapter, training_batch, journal, contract)
        donor_update, donor_ref = donor
        verify_realized_update(donor_update, donor_ref, journal)
        require(donor_update.batch_identity == training_batch.identity, "MATCHED_FORK_TRAINING_BATCH_MISMATCH")
        states = {"skip": entry.clone(), "native_full": native.post.clone(),
                  "native_parameter_only": parameter_state(entry, native.parameter_delta),
                  "native_optimizer_only": entry.clone(),
                  "donor_parameter_delta": parameter_state(entry, donor_update.parameter_delta),
                  "donor_optimizer_innovation": transport(entry, donor_update.pre, donor_update.post, adapter.optimizer_semantics())}
        states["native_optimizer_only"].optimizer = native.post.clone().optimizer
        # The historical native-only branch uses the exact post parameters, not
        # subtraction/addition reconstruction that can introduce FP32 rounding.
        states["native_parameter_only"].parameters = native.post.clone().parameters
        states["donor_joint_update"] = parameter_state(states["donor_optimizer_innovation"], donor_update.parameter_delta)
        assert_isolated(*states.values())
        state_refs = {key: journal.record("causal_fork_state", [native_ref, donor_ref],
                      {"branch": key, "state": value, "buffer_policy": "native_full_retains_actual_buffers_other_branches_hold_receiver_buffers"})
                      for key, value in states.items()}
        observations, effects, interactions, categorical, bindings = {}, {}, {}, {}, []
        for horizon in range(1, max(horizons) + 1):
            if horizon > 1:
                for key in BRANCHES:
                    adapter.restore(states[key])
                    set_opportunity(seed + horizon)
                    opportunity_ref = journal.record("matched_rng_receiving_state", [state_refs[key]],
                                                      {"seed": seed + horizon, "state": adapter.snapshot()})
                    continuation, ref = capture_update(adapter, continuation_batches[horizon - 2], journal, contract,
                                                       state_ref=opportunity_ref)
                    states[key] = continuation.post
                    state_refs[key] = journal.record("causal_branch_continuation", [state_refs[key], ref],
                                                    {"branch": key, "horizon": horizon, "state": states[key], "rng_opportunity": seed + horizon})
            if horizon not in horizons:
                continue
            probes = {}
            for key in BRANCHES:
                adapter.restore(states[key])
                probes[key] = run_csrg(adapter, evaluation_batch, journal, state_ref=state_refs[key])
                bindings.append(probes[key]["evidence_ref"])
            observations[str(horizon)] = probes
            arrays = {key: response_arrays(p) for key, p in probes.items()}
            effects[str(horizon)] = {key: {name: value - arrays["skip"][name]
                                         for name, value in arrays[key].items() if np.issubdtype(value.dtype, np.floating)}
                                     for key in BRANCHES if key != "skip"}
            phi = effects[str(horizon)]
            interactions[str(horizon)] = {name: value - phi["native_parameter_only"][name] - phi["native_optimizer_only"][name]
                                          for name, value in phi["native_full"].items()}
            categorical[str(horizon)] = {branch: {name: value != arrays["skip"][name]
                                                  for name, value in arrays[branch].items() if not np.issubdtype(value.dtype, np.floating)}
                                         for branch in BRANCHES if branch != "skip"}
        result = dict(schema="realized-update-forks-v1.1", branches=BRANCHES, horizons=horizons,
                      receiver_pre_identity=entry.identity, donor_pre_identity=donor_update.pre.identity,
                      observations=observations, branch_minus_skip=effects, interaction_residual=interactions,
                      categorical_changed_masks=categorical,
                      initial_state_ids={k: observations["1"][k]["state_identity"] for k in BRANCHES},
                      optimizer_kind=adapter.optimizer_semantics(), branch_storage_isolated=True,
                      native_buffer_changed=any(torch.count_nonzero(v).item() for v in native.buffer_delta.values()),
                      buffer_semantics="native_full=actual_parameters+buffers+optimizer; other_h1_branches=receiver_buffers",
                      classification="NOT_ADJUDICATED_WITHOUT_RECIPROCAL_SIGNATURE_ROWS")
        result["evidence_ref"] = journal.record("causal_branch_minus_skip_derivation", bindings, result)
        return result
    finally:
        adapter.restore(entry)


def signature_row(transferred, recipient_native, donor_native, *, recipient, donor, horizon, transferred_branch, native_branch):
    """Frozen normalized RMS/median signature; only storage bindings changed."""
    from statistics import median
    require(set(transferred) == set(recipient_native) == set(donor_native) and transferred, "RECIPROCAL_ROLE_SET_MISMATCH")
    rows = []
    for key in sorted(transferred):
        t, r, d = [np.asarray(v[key], dtype=np.float64) for v in (transferred, recipient_native, donor_native)]
        require(t.shape == r.shape == d.shape, "RECIPROCAL_RESPONSE_SHAPE_MISMATCH")
        finite = np.isfinite(t) & np.isfinite(r) & np.isfinite(d)
        if not finite.any():
            continue
        t, r, d = t[finite], r[finite], d[finite]
        rms = lambda value: float(np.sqrt(np.mean(np.square(value, dtype=np.float64), dtype=np.float64)))
        separation = rms(d - r)
        if separation <= 1e-15:
            continue
        dd, rd = rms(t - d) / separation, rms(t - r) / separation
        rows.append(dict(key=key, donor_distance=dd, recipient_distance=rd, donor_closer=dd < rd))
    dm = median(row["donor_distance"] for row in rows) if rows else None
    rm = median(row["recipient_distance"] for row in rows) if rows else None
    return dict(recipient=recipient, donor=donor, horizon=horizon, transferred_branch=transferred_branch,
                native_signature_branch=native_branch, informative_tensor_count=len(rows),
                donor_closer_tensor_count=sum(row["donor_closer"] for row in rows),
                median_normalized_distance_to_donor=dm, median_normalized_distance_to_recipient=rm,
                aggregate_donor_closer=dm < rm if rows else None,
                adjudication_status="INFORMATIVE" if rows else "UNINFORMATIVE_DONOR_AND_RECIPIENT_NATIVE_RESPONSES_IDENTICAL",
                per_tensor=rows)


def adjudicate_reciprocal(a, b, journal):
    require(tuple(a["horizons"]) == tuple(b["horizons"]), "RECIPROCAL_HORIZON_MISMATCH")
    require(a["receiver_pre_identity"] == b["donor_pre_identity"] and b["receiver_pre_identity"] == a["donor_pre_identity"], "RECIPROCAL_DONOR_RECEIVER_NOT_MATCHED")
    ap, bp = a["observations"]["1"]["skip"], b["observations"]["1"]["skip"]
    require(all(ap[key] == bp[key] for key in ("component_ids", "group_order", "boundary_id", "item_hashes")), "RECIPROCAL_TARGET_ALIGNMENT")
    mapping = {"donor_parameter_delta": "native_parameter_only", "donor_optimizer_innovation": "native_optimizer_only", "donor_joint_update": "native_full"}
    rows = []
    for recipient, donor, left, right in (("A", "B", a, b), ("B", "A", b, a)):
        for h in left["horizons"]:
            for branch, native in mapping.items():
                rows.append(signature_row(left["branch_minus_skip"][str(h)][branch], left["branch_minus_skip"][str(h)][native],
                                           right["branch_minus_skip"][str(h)][native], recipient=recipient, donor=donor,
                                           horizon=h, transferred_branch=branch, native_branch=native))
    verdict = (classify_reciprocal(rows) if tuple(a["horizons"]) == (1, 2, 5, 20, 100) else
               dict(decision="NOT_EVALUABLE", reason="FROZEN_RECIPROCAL_HORIZONS_NOT_COMPLETE"))
    result = dict(schema="reciprocal-adjudication-v1.1", signatures=rows, verdict=verdict)
    result["evidence_ref"] = journal.record("reciprocal_signature_adjudication", [a["evidence_ref"], b["evidence_ref"]], result)
    return result


def classify_reciprocal(rows):
    """Versioned fail-closed repair of the frozen reciprocal classifier."""
    expected = {(receiver, branch, h) for receiver in ("A", "B")
                for branch in ("donor_parameter_delta", "donor_optimizer_innovation", "donor_joint_update")
                for h in (1, 2, 5, 20, 100)}
    index = {(r["recipient"], r["transferred_branch"], int(r["horizon"])): r["aggregate_donor_closer"] for r in rows}
    require(len(index) == len(rows) and set(index) == expected, "RECIPROCAL_SIGNATURE_COVERAGE")
    require(all(v is None or type(v) is bool for v in index.values()), "INVALID_RECIPROCAL_TRISTATE")
    if any(v is None for v in index.values()):
        return {"decision": "NOT_EVALUABLE", "reason": "UNINFORMATIVE_RECIPROCAL_SIGNATURE",
                "semantics": "reciprocal-classifier-v1.1"}
    def both(branch, h):
        return index["A", branch, h] and index["B", branch, h]
    parameter_h1 = both("donor_parameter_delta", 1)
    optimizer_h1 = both("donor_optimizer_innovation", 1)
    joint_h1 = both("donor_joint_update", 1)
    later = [h for h in (2, 5, 20, 100) if both("donor_optimizer_innovation", h) or both("donor_joint_update", h)]
    any_bilateral = any(both(b, h) for b in ("donor_parameter_delta", "donor_optimizer_innovation", "donor_joint_update") for h in (1, 2, 5, 20, 100))
    decision = ("UPDATE_DOMINANT" if parameter_h1 and optimizer_h1 and joint_h1 else
                "DELAYED_OPTIMIZER_DOMINANT" if parameter_h1 and not optimizer_h1 and later else
                "RECEIVING_STATE_DOMINANT" if not any_bilateral else
                "NONSEPARABLE_INTERACTION_OR_OBSERVATION_FAILURE")
    return {"decision": decision, "semantics": "reciprocal-classifier-v1.1"}
