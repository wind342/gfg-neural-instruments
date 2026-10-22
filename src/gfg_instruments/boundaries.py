"""Task readouts are explicit contracts; support formulas never live in adapters."""
from __future__ import annotations

import numpy as np
import torch

from .contracts import Batch, BoundaryContract
from .errors import require


def readout(output, targets, batch: Batch, contract: BoundaryContract):
    values = output.detach().cpu().numpy()
    require(values.shape[0] == len(batch.identities), "READOUT_IDENTITY_COUNT_MISMATCH")
    require(np.isfinite(values).all(), "NONFINITE_NATIVE_OUTPUT")
    if contract.kind == "classification":
        labels = targets.detach().cpu().numpy()
        require(values.ndim == 2 and values.shape[1] >= 2 and labels.shape == (len(values),), "CLASSIFICATION_SHAPE_INVALID")
        require(np.issubdtype(labels.dtype, np.integer), "TARGET_LABEL_NOT_INTEGER")
        require(((labels >= 0) & (labels < values.shape[1])).all(), "TARGET_LABEL_OUT_OF_RANGE")
        masked = values.copy()
        rows = np.arange(len(values))
        masked[rows, labels] = -np.inf
        margins = values[rows, labels] - masked.max(axis=1)
        predictions = values.argmax(axis=1)
        correct = predictions == labels
    else:
        truth = targets.detach().cpu().numpy()
        require("candidates" in batch.extras, "FROZEN_RESIDUAL_CANDIDATES_REQUIRED")
        candidates = batch.extras["candidates"].detach().cpu().numpy()
        require(values.shape == truth.shape and candidates.shape[0] == len(values)
                and candidates.shape[2:] == values.shape[1:] and candidates.shape[1] > 0,
                "RESIDUAL_CANDIDATE_SHAPE_INVALID")
        require(np.isfinite(truth).all() and np.isfinite(candidates).all(), "NONFINITE_RESIDUAL_TARGET")
        true_error = ((values - truth) ** 2).reshape(len(values), -1).mean(axis=1)
        candidate_errors = ((values[:, None] - candidates) ** 2).reshape(len(values), candidates.shape[1], -1).mean(axis=2)
        margins = (candidate_errors - true_error[:, None]).min(axis=1)
        predictions = np.concatenate([true_error[:, None], candidate_errors], axis=1).argmin(axis=1)
        correct = margins >= 0
    if contract.correct_rule == "positive":
        correct = margins > 0
    elif contract.correct_rule == "nonnegative":
        correct = margins >= 0
    require(np.isfinite(margins).all(), "NONFINITE_MARGIN")
    group_order = tuple(sorted(set(batch.groups)))
    membership = np.asarray(batch.groups)
    # Same linear-interpolation q10 as the frozen nanoGPT CSRG contract.
    q10 = np.asarray([np.quantile(margins[membership == g], .10, method="linear")
                      for g in group_order], dtype=np.float64)
    return dict(output=output.detach().cpu().clone(), margins=margins, predictions=predictions,
                correct=correct, group_membership=batch.groups, group_order=group_order,
                group_q10_margin=q10, identities=batch.identities, item_hashes=batch.item_hashes(),
                targets=targets.detach().cpu().clone(), extras=batch.extras, boundary=contract.__dict__,
                boundary_id=contract.identity)


def validate_readout(row):
    """Independent recomputation from the persisted native output and targets."""
    contract = BoundaryContract(**row["boundary"])
    require(contract.identity == row["boundary_id"], "READOUT_CONTRACT_HASH_MISMATCH")
    values = row["output"].numpy()
    labels = row["targets"].numpy()
    require(np.isfinite(values).all(), "NONFINITE_RECORDED_OUTPUT")
    margins, predictions, correctness = [], [], []
    for i, output in enumerate(values):
        if contract.kind == "classification":
            target = int(labels[i])
            require(labels[i] == target and 0 <= target < len(output), "RECORDED_TARGET_INVALID")
            competitors = np.delete(output, target)
            margin = output[target] - competitors.max()
            predicted = int(output.argmax())
            correct = predicted == target
        else:
            candidates = row["extras"]["candidates"][i].numpy()
            true_error = np.mean((output - labels[i]) ** 2)
            errors = np.asarray([np.mean((output - c) ** 2) for c in candidates])
            margin = (errors - true_error).min()
            predicted = int(np.r_[true_error, errors].argmin())
            correct = margin >= 0
        if contract.correct_rule == "positive":
            correct = margin > 0
        elif contract.correct_rule == "nonnegative":
            correct = margin >= 0
        margins.append(margin)
        predictions.append(predicted)
        correctness.append(correct)
    require(len(row["identities"]) == len(set(row["identities"])) == len(values)
            and set(row["identities"]) == set(row["item_hashes"]), "RECORDED_IDENTITY_COVERAGE")
    margins = np.asarray(margins)
    require(np.array_equal(margins, row["margins"]) and np.array_equal(predictions, row["predictions"])
            and np.array_equal(correctness, row["correct"]), "READOUT_RECOMPUTATION_FAILED")
    membership = np.asarray(row["group_membership"])
    require(tuple(row["group_order"]) == tuple(sorted(set(membership))), "READOUT_GROUP_ORDER")
    q10 = np.asarray([np.quantile(margins[membership == group], .1, method="linear") for group in row["group_order"]])
    require(np.array_equal(q10, row["group_q10_margin"]), "READOUT_Q10_RECOMPUTATION_FAILED")
    return {"status": "PASS", "targets": len(values)}
