"""Standalone, read-only artifact and scientific-result recomputation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .core_bridge import validate_core_export
from .csrg import validate_csrg
from .errors import require
from .evidence import payload, validate_journal
from .paths import validate_path
from .boundaries import validate_readout
from .identity import digest


def verify_result(root):
    root = Path(root)
    document = validate_journal(root)
    counts = {"csrg": 0, "paths": 0, "forks": 0, "native_updates": 0, "ledgers": 0}
    events = {row["id"]: row for row in document["events"]}
    for row in document["events"]:
        operation = row["operation"]
        if operation == "native_component_gate_forward":
            validate_readout(payload(root, row["artifact"]))
        if operation == "csrg4c_scientific_derivation":
            validate_csrg(payload(root, row["artifact"]))
            counts["csrg"] += 1
        elif operation == "finite_amplitude_completed":
            validate_path(payload(root, row["artifact"]))
            counts["paths"] += 1
        elif operation == "causal_branch_minus_skip_derivation":
            result = payload(root, row["artifact"])
            from .forks import response_arrays
            for horizon, branches in result["branch_minus_skip"].items():
                skip = response_arrays(result["observations"][horizon]["skip"])
                for key, fields in branches.items():
                    actual = response_arrays(result["observations"][horizon][key])
                    for name, value in fields.items():
                        require(np.array_equal(value, actual[name] - skip[name], equal_nan=True), "FORK_RESPONSE_RECOMPUTATION_FAILED")
                for name, value in result["interaction_residual"][horizon].items():
                    expected = branches["native_full"][name] - branches["native_parameter_only"][name] - branches["native_optimizer_only"][name]
                    require(np.array_equal(value, expected, equal_nan=True), "FORK_INTERACTION_RECOMPUTATION_FAILED")
                for branch, masks in result["categorical_changed_masks"][horizon].items():
                    actual = response_arrays(result["observations"][horizon][branch])
                    for name, mask in masks.items():
                        require(np.array_equal(mask, actual[name] != skip[name]), "FORK_CATEGORY_RECOMPUTATION_FAILED")
            counts["forks"] += 1
        elif operation == "actual_native_optimizer_step":
            update = payload(root, row["artifact"])
            for name, delta in update["parameter_delta"].items():
                require(torch.equal(delta, update["post"]["parameters"][name] - update["pre"]["parameters"][name]), "NATIVE_PARAMETER_DELTA_MISMATCH")
            for name, delta in update["buffer_delta"].items():
                require(torch.equal(delta, update["post"]["buffers"][name] - update["pre"]["buffers"][name]), "NATIVE_BUFFER_DELTA_MISMATCH")
            counts["native_updates"] += 1
        elif operation == "identity_aligned_boundary_adjudication":
            result = payload(root, row["artifact"])
            seal_event, actual_event = [events[key] for key in row["parents"]]
            require(seal_event["operation"] == "pretarget_prediction_seal" and seal_event["index"] < actual_event["index"] < row["index"], "LEDGER_EVENT_ORDER_INVALID")
            seal, actual = payload(root, seal_event["artifact"]), payload(root, actual_event["artifact"])
            require(digest({k: v for k, v in seal.items() if k != "seal_sha256"}) == seal["seal_sha256"], "LEDGER_SEAL_HASH_MISMATCH")
            require(seal["item_hashes"] == actual["item_hashes"] and seal["boundary_id"] == actual["boundary_id"], "LEDGER_IDENTITY_CHANGED")
            index = {key: i for i, key in enumerate(actual["identities"])}
            require(set(index) == set(seal["identities"]), "LEDGER_IDENTITY_SET_CHANGED")
            for i, item in enumerate(result["rows"]):
                require(item["identity"] == seal["identities"][i], "LEDGER_ROW_MISALIGNED")
                if item["status"] == "EVALUABLE":
                    truth = bool(actual["correct"][index[item["identity"]]])
                    predicted = bool(seal["predicted_correct"][i])
                    require(item["actual"] == truth and item["predicted"] == predicted
                            and item["prediction_matches"] == (truth == predicted), "LEDGER_VERDICT_MISMATCH")
                else:
                    require(item["predicted"] is None and item["actual"] is None and item["prediction_matches"] is None, "UNDEFINED_VERDICT_BECAME_FALSE")
            evaluated = [item for item in result["rows"] if item["status"] == "EVALUABLE"]
            expected = sum(item["prediction_matches"] for item in evaluated) / len(evaluated) if evaluated else None
            require(expected == result["accuracy"] and len(evaluated) == result["evaluable_count"], "LEDGER_AGGREGATE_MISMATCH")
            for group, capability in result["group_capabilities"].items():
                members = [r for r in result["rows"] if seal["group_by_identity"][r["identity"]] == group]
                require(len(members) == capability["count"], "LEDGER_GROUP_COUNT_MISMATCH")
                if capability["status"] == "EVALUABLE":
                    before = float(np.mean([r["before"] for r in members]))
                    for key, field in (("predicted", "predicted"), ("actual", "actual")):
                        value = float(np.mean([r[field] for r in members]))
                        require(value == capability[key] and value - before == capability[key + "_delta"]
                                and int(np.sign(value - before)) == capability[key + "_direction"], "LEDGER_CAPABILITY_MISMATCH")
            counts["ledgers"] += 1
    core = validate_core_export(root)
    return {"status": "PASS", "artifact_and_receipt_hashes": "PASS", "scientific_recomputation": counts,
            "core_v3": core, "historical_full_replay": "NOT_RUN_HISTORICAL_PAYLOADS_NOT_BUNDLED"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_result(args.root), indent=2))


if __name__ == "__main__":
    main()
