from __future__ import annotations

import numpy as np

from .amplitude_kernel import AMPLITUDE_SCALES, RESPONSE_CENTERS, TensorStore, _numeric_path, scale_key
from .contracts import ExecutionContract
from .csrg import run_csrg
from .errors import require
from .forks import capture_update, parameter_state, response_arrays, set_opportunity, verify_realized_update
from .identity import digest


def run_amplitude_path(adapter, batch, journal, *, realized_update, horizons=(1,),
                       continuation_batches=(), training_contract=ExecutionContract(), seed=2021):
    """Frozen TL-E06 eleven-scale/five-center kernel. Initial buffers and memory held.

    This parameter-only path is intentionally NOT the historical ResNet combined
    parameter+BN-buffer path. Native continuation subsequently updates both.
    """
    require(horizons and tuple(sorted(set(horizons))) == tuple(horizons) and horizons[0] == 1, "INVALID_PATH_HORIZONS")
    require(len(continuation_batches) == max(horizons) - 1, "PATH_CONTINUATION_COVERAGE")
    update, update_ref = realized_update
    verify_realized_update(update, update_ref, journal)
    entry = adapter.snapshot()
    receiver_ref = journal.source("receiving-state:" + entry.identity, entry)
    contract = dict(scales=AMPLITUDE_SCALES, centers=RESPONSE_CENTERS, epsilon=.125,
                    horizons=horizons, scope="parameters_only", optimizer="held_at_receiver_initially",
                    buffers="held_at_receiver_initially", continuation="native_real_updates")
    contract_ref = journal.source("amplitude-contract:" + digest(contract), contract)
    probes, arrays, states, refs, continuations = {}, {}, {}, {}, {}
    try:
        for scale in AMPLITUDE_SCALES:
            state = parameter_state(entry, update.parameter_delta, scale)
            adapter.restore(state)
            ref = journal.record("realized_parameter_amplitude_state", [receiver_ref, update_ref, contract_ref],
                                 {"scale": scale, "state": state})
            probe = run_csrg(adapter, batch, journal, state_ref=ref)
            probes[scale_key(scale)] = probe
            arrays[scale] = response_arrays(probe)
            if scale in RESPONSE_CENTERS:
                states[scale], refs[scale] = state, ref
        numeric, categorical = _numeric_path(TensorStore(), label="receiver", arrays=arrays, epsilon=.125)
        response = dict(contract=contract, numeric=numeric, categorical=categorical)
        response_ref = journal.record("frozen_five_center_numeric_and_categorical_path",
                                      [p["evidence_ref"] for p in probes.values()], response)
        # Numeric/categorical response is materialized before any future training.
        for horizon in range(2, max(horizons) + 1):
            for scale in RESPONSE_CENTERS:
                adapter.restore(states[scale])
                set_opportunity(seed + horizon)
                opportunity_ref = journal.record("matched_rng_receiving_state", [refs[scale]],
                                                  {"seed": seed + horizon, "state": adapter.snapshot()})
                follow, ref = capture_update(adapter, continuation_batches[horizon - 2], journal, training_contract,
                                             state_ref=opportunity_ref)
                states[scale] = follow.post
                refs[scale] = journal.record("amplitude_path_native_continuation", [refs[scale], ref, response_ref],
                                            {"scale": scale, "horizon": horizon, "state": follow.post,
                                             "rng_opportunity": seed + horizon})
                if horizon in horizons:
                    continuations[f"{horizon}:{scale_key(scale)}"] = run_csrg(adapter, batch, journal, state_ref=refs[scale])
        result = dict(schema="finite-amplitude-parameter-path-v1", contract=contract, probes=probes,
                      numeric=numeric, categorical=categorical, continuations=continuations,
                      response_frozen_before_continuation=True,
                      native_update_buffer_effects_excluded=True)
        result["evidence_ref"] = journal.record("finite_amplitude_completed",
                                               [response_ref, *[p["evidence_ref"] for p in continuations.values()]], result)
        return result
    finally:
        adapter.restore(entry)


def finite_rms(values):
    finite = np.isfinite(values)
    return None if not finite.any() else float(np.sqrt(np.mean(np.asarray(values)[finite].astype(np.float64) ** 2)))


def validate_path(result):
    require(tuple(result["contract"]["scales"]) == AMPLITUDE_SCALES, "FROZEN_SCALE_GRID_CHANGED")
    require(result["response_frozen_before_continuation"], "PATH_RESPONSE_NOT_PRECONTINUATION")
    for role, row in result["numeric"].items():
        values = {scale: response_arrays(result["probes"][scale_key(scale)])[role].astype(np.float64) for scale in AMPLITUDE_SCALES}
        first = []
        for center in RESPONSE_CENTERS:
            j = (values[center + .125] - values[center - .125]) / .25
            k = (values[center + .125] + values[center - .125] - 2 * values[center]) / (.125 ** 2)
            require(np.array_equal(j, row["centers"][scale_key(center)]["j_first_order"], equal_nan=True), "PATH_J_MISMATCH")
            require(np.array_equal(k, row["centers"][scale_key(center)]["k_curvature"], equal_nan=True), "PATH_K_MISMATCH")
            first.append(j)
        expected = (first[0] + 4 * first[1] + 2 * first[2] + 4 * first[3] + first[4]) / 12
        require(np.array_equal(expected, row["simpson_delta_prediction"], equal_nan=True), "PATH_SIMPSON_MISMATCH")
        require(np.array_equal(values[1.] - values[0.], row["exact_scale_zero_to_one_delta"], equal_nan=True), "PATH_ENDPOINT_MISMATCH")
    for role, row in result["categorical"].items():
        values = {scale: response_arrays(result["probes"][scale_key(scale)])[role] for scale in AMPLITUDE_SCALES}
        require(np.array_equal(values[1.] != values[0.], row["endpoint_changed_mask"]), "CATEGORY_ENDPOINT_MISMATCH")
        for left, right in zip(AMPLITUDE_SCALES, AMPLITUDE_SCALES[1:]):
            observed = row["adjacent_transitions"][f"{scale_key(left)}_to_{scale_key(right)}"]
            mask = values[left] != values[right]
            require(np.array_equal(mask, observed["changed_mask"]) and int(mask.sum()) == observed["changed_count"], "CATEGORY_PATH_MISMATCH")
    return {"status": "PASS", "numeric_roles": len(result["numeric"]), "categorical_roles": len(result["categorical"])}
