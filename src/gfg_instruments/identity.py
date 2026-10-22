from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json

import numpy as np
import torch


def tensor_identity(value):
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().contiguous()
        raw = data.reshape(-1).view(torch.uint8).numpy().tobytes()
        dtype = str(data.dtype)
    else:
        data = np.ascontiguousarray(value)
        raw = data.tobytes()
        dtype = str(data.dtype)
    return {"dtype": dtype, "shape": list(data.shape), "sha256": hashlib.sha256(raw).hexdigest()}


def normalized(value):
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return {"tensor": tensor_identity(value)}
    if isinstance(value, np.generic):
        return normalized(value.item())
    if is_dataclass(value):
        return normalized(asdict(value))
    if isinstance(value, dict):
        return {str(key): normalized(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalized(child) for child in value]
    return value


def canonical(value):
    return json.dumps(normalized(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()
