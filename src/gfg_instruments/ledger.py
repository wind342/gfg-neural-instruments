"""Identity-aligned prediction seal and read-only endpoint adjudication."""
from __future__ import annotations

import numpy as np

from .csrg import measure
from .errors import require
from .forks import parameter_state, verify_realized_update
from .identity import digest


def seal_prediction(batch, boundary, before, predicted_margin, journal, parents, *, method):
    values = np.asarray(predicted_margin, dtype=np.float64)
    require(values.shape == (len(batch.identities),), "LEDGER_PREDICTION_SHAPE")
    require(before["item_hashes"] == batch.item_hashes() and before["boundary_id"] == boundary.identity, "LEDGER_PRESTATE_IDENTITY_MISMATCH")
    evaluable = np.isfinite(values)
    if not boundary.zero_evaluable:
        evaluable &= (values != 0) & (before["margins"] != 0)
    seal = dict(schema="identity-boundary-ledger-v1", identities=batch.identities,
                group_by_identity=dict(zip(batch.identities, batch.groups)),
                item_hashes=batch.item_hashes(), boundary_id=boundary.identity,
                before_correct=np.asarray(before["correct"], dtype=bool), before_margins=before["margins"],
                predicted_margins=values, evaluable=evaluable,
                predicted_correct=values >= 0 if boundary.correct_rule == "nonnegative" else values > 0,
                method=method, post_update_endpoint_read=False)
    seal["seal_sha256"] = digest(seal)
    ref = journal.record("pretarget_prediction_seal", parents, seal)
    return {**seal, "evidence_ref": ref}


def adjudicate(seal, actual, batch, boundary, journal, actual_ref):
    require(digest({k: v for k, v in seal.items() if k not in ("seal_sha256", "evidence_ref")}) == seal["seal_sha256"], "PREDICTION_SEAL_CHANGED")
    require(seal["evidence_ref"] in journal._known and actual_ref in journal._known, "LEDGER_UNBOUND_RESULT")
    events = {r["id"]: r for r in journal.events}
    require(seal["evidence_ref"] in events and actual_ref in events
            and events[seal["evidence_ref"]]["index"] < events[actual_ref]["index"], "LEDGER_ENDPOINT_MUST_FOLLOW_SEAL")
    require(journal.artifact(actual) == events[actual_ref]["artifact"], "LEDGER_ENDPOINT_RECEIPT_MISMATCH")
    order = {key: i for i, key in enumerate(actual["identities"])}
    require(len(order) == len(seal["identities"]) and set(order) == set(seal["identities"]), "LEDGER_TARGET_SET_MISMATCH")
    require(seal["item_hashes"] == batch.item_hashes() == actual["item_hashes"], "LEDGER_TARGET_CONTENT_MISMATCH")
    require(seal["boundary_id"] == boundary.identity == actual["boundary_id"], "LEDGER_BOUNDARY_CHANGED")
    rows = []
    for i, identity in enumerate(seal["identities"]):
        j = order[identity]
        valid = bool(seal["evaluable"][i] and np.isfinite(actual["margins"][j]) and
                     (boundary.zero_evaluable or actual["margins"][j] != 0))
        before, predicted, after = bool(seal["before_correct"][i]), bool(seal["predicted_correct"][i]), bool(actual["correct"][j])
        rows.append(dict(identity=identity, status="EVALUABLE" if valid else "NOT_EVALUABLE",
                         before=before, predicted=predicted if valid else None, actual=after if valid else None,
                         transition=("correct" if before else "wrong") + "->" + ("correct" if after else "wrong") if valid else None,
                         prediction_matches=(predicted == after) if valid else None))
    valid_rows = [r for r in rows if r["status"] == "EVALUABLE"]
    capabilities = {}
    for group in sorted(set(seal["group_by_identity"].values())):
        members = [r for r in rows if seal["group_by_identity"][r["identity"]] == group]
        evaluable = all(r["status"] == "EVALUABLE" for r in members)
        before = float(np.mean([r["before"] for r in members]))
        predicted = float(np.mean([r["predicted"] for r in members])) if evaluable else None
        actual_capability = float(np.mean([r["actual"] for r in members])) if evaluable else None
        capabilities[group] = dict(status="EVALUABLE" if evaluable else "NOT_EVALUABLE", count=len(members),
            before=before, predicted=predicted, actual=actual_capability,
            predicted_delta=predicted - before if evaluable else None,
            actual_delta=actual_capability - before if evaluable else None,
            predicted_direction=int(np.sign(predicted - before)) if evaluable else None,
            actual_direction=int(np.sign(actual_capability - before)) if evaluable else None)
    result = dict(schema="identity-boundary-adjudication-v1", seal_sha256=seal["seal_sha256"], rows=rows,
                  group_capabilities=capabilities,
                  evaluable_count=len(valid_rows), not_evaluable_count=len(rows) - len(valid_rows),
                  accuracy=sum(r["prediction_matches"] for r in valid_rows) / len(valid_rows) if valid_rows else None,
                  post_endpoint_read_after_seal=True)
    result["evidence_ref"] = journal.record("identity_aligned_boundary_adjudication", [seal["evidence_ref"], actual_ref], result)
    for row in rows:
        if row["status"] == "NOT_EVALUABLE":
            journal.record("undefined_boundary_verdict", [result["evidence_ref"]], {"identity": row["identity"]}, disposition="NONFINITE_OR_EXACT_ZERO_MARGIN")
    return result


def run_boundary_ledger(adapter, batch, journal, *, realized_update):
    """Original deterministic m+J+K/2 readout, NOT a replacement TL-P01 predictor.

    J/K use the frozen epsilon .125 central finite differences. The alpha=1
    target is not evaluated until the prediction has been recorded and hashed.
    """
    update, update_ref = realized_update
    verify_realized_update(update, update_ref, journal)
    entry = adapter.snapshot()
    receiver = journal.source("receiving-state:" + entry.identity, entry)
    observations, refs = {}, []
    try:
        for scale in (0., -.125, .125):
            state = parameter_state(entry, update.parameter_delta, scale)
            adapter.restore(state)
            state_ref = journal.record("pretarget_directional_state", [receiver, update_ref], {"scale": scale, "state": state})
            observations[scale], ref = measure(adapter, batch, journal, state_ref=state_ref)
            refs.append(ref)
        base = observations[0.]["margins"].astype(np.float64)
        plus = observations[.125]["margins"].astype(np.float64)
        minus = observations[-.125]["margins"].astype(np.float64)
        j = (plus - minus) / .25
        k = (plus + minus - 2 * base) / (.125 ** 2)
        seal = seal_prediction(batch, adapter.boundary, observations[0.], base + j + .5 * k,
                               journal, refs, method="frozen-central-difference-margin-jk-v1")
        endpoint = parameter_state(entry, update.parameter_delta, 1.)
        adapter.restore(endpoint)
        state_ref = journal.record("postseal_parameter_endpoint", [receiver, update_ref, seal["evidence_ref"]], endpoint)
        actual, ref = measure(adapter, batch, journal, state_ref=state_ref)
        result = adjudicate(seal, actual, batch, adapter.boundary, journal, ref)
        return {"seal": seal, "adjudication": result, "state_preserved": True}
    finally:
        adapter.restore(entry)
