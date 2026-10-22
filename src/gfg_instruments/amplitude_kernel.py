from __future__ import annotations

from typing import Any
import numpy as np
from .errors import require

# Function body copied verbatim from frozen TL-E06 amplitude_path.py.
AMPLITUDE_SCALES = (-.125, 0., .125, .25, .375, .5, .625, .75, .875, 1., 1.125)
RESPONSE_CENTERS = (0., .25, .5, .75, 1.)

def scale_key(scale):
    sign = "m" if scale < 0 else "p"
    return sign + f"{abs(scale):.3f}".replace(".", "p")

class TensorStore:
    def put(self, value, *, representation):
        return value.copy()

def _numeric_path(
    store: TensorStore,
    *,
    label: str,
    arrays: dict[float, dict[str, np.ndarray]],
    epsilon: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    keys = tuple(sorted(arrays[0.0]))
    require(
        all(tuple(sorted(value)) == keys for value in arrays.values()),
        "SST_AMPLITUDE_PATH_PROBE_OUTPUT_SET_MISMATCH",
    )
    numeric: dict[str, Any] = {}
    categorical: dict[str, Any] = {}
    for key in keys:
        values = {scale: arrays[scale][key] for scale in AMPLITUDE_SCALES}
        shapes = {tuple(value.shape) for value in values.values()}
        require(len(shapes) == 1, f"SST_AMPLITUDE_PATH_SHAPE_MISMATCH:{key}")
        prefix = f"amplitude-path:{label}:h1:{key}"
        if np.issubdtype(values[0.0].dtype, np.floating):
            values64 = {
                scale: value.astype(np.float64, copy=False)
                for scale, value in values.items()
            }
            centers: dict[str, Any] = {}
            derivatives: dict[float, tuple[np.ndarray, np.ndarray]] = {}
            for center in RESPONSE_CENTERS:
                plus = values64[center + epsilon]
                base = values64[center]
                minus = values64[center - epsilon]
                j_value = (plus - minus) / (2.0 * epsilon)
                k_value = (plus + minus - 2.0 * base) / (epsilon * epsilon)
                derivatives[center] = (j_value, k_value)
                center_key = scale_key(center)
                centers[center_key] = {
                    "scale": center,
                    "j_first_order": store.put(
                        j_value,
                        representation=f"{prefix}:{center_key}:central_first_order_response",
                    ),
                    "k_curvature": store.put(
                        k_value,
                        representation=f"{prefix}:{center_key}:central_second_order_curvature",
                    ),
                }
            j0, k0 = derivatives[0.0]
            j1, k1 = derivatives[1.0]
            simpson = (
                derivatives[0.0][0]
                + 4.0 * derivatives[0.25][0]
                + 2.0 * derivatives[0.5][0]
                + 4.0 * derivatives[0.75][0]
                + derivatives[1.0][0]
            ) / 12.0
            exact = values64[1.0] - values64[0.0]
            numeric[key] = {
                "scale_values": {
                    scale_key(scale): {
                        "scale": scale,
                        "value": store.put(
                            value,
                            representation=f"{prefix}:{scale_key(scale)}:numeric_value_float64",
                        ),
                    }
                    for scale, value in values64.items()
                },
                "centers": centers,
                "simpson_delta_prediction": store.put(
                    simpson,
                    representation=f"{prefix}:fixed_five_center_simpson_delta_prediction",
                ),
                "exact_scale_zero_to_one_delta": store.put(
                    exact,
                    representation=f"{prefix}:exact_scale_zero_to_one_delta",
                ),
                "start_endpoint_j_prediction": store.put(
                    j0,
                    representation=f"{prefix}:start_endpoint_first_order_prediction",
                ),
                "start_endpoint_jk_prediction": store.put(
                    j0 + 0.5 * k0,
                    representation=f"{prefix}:start_endpoint_signed_second_order_prediction",
                ),
                "end_endpoint_j_prediction": store.put(
                    j1,
                    representation=f"{prefix}:end_endpoint_first_order_prediction",
                ),
                "end_endpoint_jk_prediction": store.put(
                    j1 - 0.5 * k1,
                    representation=f"{prefix}:end_endpoint_signed_second_order_prediction",
                ),
            }
        else:
            baseline = values[0.0]
            endpoint = values[1.0]
            scale_values = {
                scale_key(scale): {
                    "scale": scale,
                    "value": store.put(
                        value,
                        representation=f"{prefix}:{scale_key(scale)}:categorical_value",
                    ),
                }
                for scale, value in values.items()
            }
            adjacent = {}
            ordered = list(AMPLITUDE_SCALES)
            for left, right in zip(ordered, ordered[1:]):
                mask = np.not_equal(values[right], values[left])
                adjacent[f"{scale_key(left)}_to_{scale_key(right)}"] = {
                    "from_scale": left,
                    "to_scale": right,
                    "changed_mask": store.put(
                        mask,
                        representation=(
                            f"{prefix}:{scale_key(left)}_to_{scale_key(right)}:"
                            "categorical_transition_mask"
                        ),
                    ),
                    "changed_count": int(np.count_nonzero(mask)),
                }
            categorical[key] = {
                "scale_values": scale_values,
                "adjacent_transitions": adjacent,
                "endpoint_changed_mask": store.put(
                    np.not_equal(endpoint, baseline),
                    representation=f"{prefix}:scale_zero_to_one_categorical_transition_mask",
                ),
                "endpoint_changed_count": int(np.count_nonzero(endpoint != baseline)),
            }
    return numeric, categorical

