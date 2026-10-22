"""Separate validated optimizer-memory transports; no generic Adam fallback."""
from __future__ import annotations

import torch

from .errors import require
from .identity import digest
from .state import Snapshot


def kind(optimizer):
    supported = {torch.optim.Adam: "adam", torch.optim.AdamW: "adamw", torch.optim.SGD: "sgd_momentum"}
    require(type(optimizer) in supported, "UNSUPPORTED_OPTIMIZER")
    result = supported[type(optimizer)]
    for group in optimizer.param_groups:
        if result != "sgd_momentum":
            require(not group.get("amsgrad", False), "AMSGRAD_TRANSPORT_NOT_VALIDATED")
        else:
            require(group["momentum"] > 0, "MOMENTUM_REQUIRED_FOR_MEMORY_FORK")
        require(not group.get("differentiable", False), "DIFFERENTIABLE_OPTIMIZER_NOT_VALIDATED")
    return result


def _groups(state):
    require(len(state.optimizer_names) == len(state.optimizer["param_groups"]), "OPTIMIZER_GROUP_LAYOUT_MISMATCH")
    for names, group in zip(state.optimizer_names, state.optimizer["param_groups"]):
        require(len(names) == len(group["params"]), "OPTIMIZER_GROUP_LENGTH_MISMATCH")
        yield names, group


def transport(recipient: Snapshot, donor_pre: Snapshot, donor_post: Snapshot, optimizer_kind: str):
    require(optimizer_kind in ("adam", "adamw", "sgd_momentum"), "UNSUPPORTED_OPTIMIZER_TRANSPORT")
    require(recipient.optimizer_kind == donor_pre.optimizer_kind == donor_post.optimizer_kind, "CROSS_OPTIMIZER_TRANSPORT_FORBIDDEN")
    require(recipient.optimizer_names == donor_pre.optimizer_names == donor_post.optimizer_names, "OPTIMIZER_NAME_ALIGNMENT_FAILURE")
    result = recipient.clone()
    for (_, rg), (_, dg), (_, pg) in zip(_groups(result), _groups(donor_pre), _groups(donor_post)):
        # No transport across unequal hyperparameters or parameter groups.
        require(digest({k: v for k, v in rg.items() if k != "params"}) ==
                digest({k: v for k, v in dg.items() if k != "params"}) ==
                digest({k: v for k, v in pg.items() if k != "params"}), "OPTIMIZER_CONFIG_MISMATCH")
        for ri, di, pi in zip(rg["params"], dg["params"], pg["params"]):
            r = result.optimizer["state"].get(ri)
            d = donor_pre.optimizer["state"].get(di)
            p = donor_post.optimizer["state"].get(pi)
            require(r is not None and d is not None and p is not None, "UNINITIALIZED_MEMORY_TRANSPORT_UNDEFINED")
            if optimizer_kind in ("adam", "adamw"):
                require(set(r) == set(d) == set(p) == {"step", "exp_avg", "exp_avg_sq"}, "ADAM_MEMORY_LAYOUT_UNSUPPORTED")
                r["step"] += p["step"] - d["step"]
                for key, beta in zip(("exp_avg", "exp_avg_sq"), dg["betas"]):
                    # Unchanged RECIPROCAL-MATCHED-PAIR-v2 innovation definition.
                    r[key] = beta * r[key] + (p[key] - beta * d[key])
                require((r["exp_avg_sq"] >= 0).all(), "NEGATIVE_SECOND_MOMENT")
            else:
                require(set(r) == set(d) == set(p) == {"momentum_buffer"}, "SGD_MEMORY_LAYOUT_UNSUPPORTED")
                mu = dg["momentum"]
                r["momentum_buffer"] = mu * r["momentum_buffer"] + (p["momentum_buffer"] - mu * d["momentum_buffer"])
            require(all(torch.isfinite(v).all() for v in r.values()), "NONFINITE_TRANSPORTED_MEMORY")
    return result


def component_loads(snapshot, components, optimizer_kind):
    by_name = {}
    for names, group in _groups(snapshot):
        by_name.update({n: snapshot.optimizer["state"].get(i) for n, i in zip(names, group["params"])})

    def rms(values):
        require(values and sum(v.numel() for v in values) > 0, "EMPTY_COMPONENT_LOAD")
        return (sum(float(v.double().square().sum()) for v in values) / sum(v.numel() for v in values)) ** .5

    result = {}
    for component in components:
        names = component.parameter_names
        require(names and all(n in snapshot.parameters for n in names), "INVALID_COMPONENT_PARAMETER_MAPPING")
        row = {"parameter_rms": rms([snapshot.parameters[n] for n in names])}
        if any(by_name.get(n) is None for n in names):
            row.update(status="UNDEFINED", reason="OPTIMIZER_MEMORY_NOT_INITIALIZED_OR_NOT_OWNED")
        elif optimizer_kind in ("adam", "adamw"):
            row.update(status="DEFINED", exp_avg_rms=rms([by_name[n]["exp_avg"] for n in names]),
                       preconditioned_rms=rms([by_name[n]["exp_avg"] / (torch.sqrt(by_name[n]["exp_avg_sq"]) + 1e-8) for n in names]),
                       preconditioned_measurement_epsilon=1e-8,
                       exp_avg_sq_sqrt_mean=(sum(float(by_name[n]["exp_avg_sq"].double().sum()) for n in names)
                                             / sum(by_name[n]["exp_avg_sq"].numel() for n in names)) ** .5)
        elif optimizer_kind == "sgd_momentum":
            row.update(status="DEFINED", momentum_rms=rms([by_name[n]["momentum_buffer"] for n in names]))
        else:
            require(False, "UNSUPPORTED_OPTIMIZER_LOAD")
        result[component.identity] = row
    return result
