"""Small real executions, not a rerun of historical scientific experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch

from .adapters.nanogpt import GPT, GPTConfig, NanoGPTAdapter
from .adapters.resnet import CifarResNet18, ResNetAdapter
from .contracts import Batch, ExecutionContract
from .core_bridge import export_core, instrument_code_hashes
from .csrg import run_csrg
from .evidence import Journal
from .forks import capture_update, run_forks
from .ledger import run_boundary_ledger
from .paths import run_amplitude_path
from .verify import verify_result


def make_case(name, device):
    generator = torch.Generator().manual_seed(20211022)
    if name == "nanogpt":
        model = GPT(GPTConfig(block_size=3, vocab_size=9, n_layer=2, n_head=2, n_embd=16, dropout=0., bias=False)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, betas=(.9, .98), weight_decay=.1, foreach=False)
        adapter = NanoGPTAdapter(model, optimizer)
        inputs = torch.randint(0, 9, (12, 3), generator=generator)
        targets = torch.full_like(inputs, -1)
        targets[:, -1] = (inputs[:, 0] + inputs[:, 1]) % 9
        groups = tuple(f"target:{int(v):02d}" for v in targets[:, -1])
    else:
        model = CifarResNet18(num_classes=5).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=.001)
        adapter = ResNetAdapter(model, optimizer)
        inputs = torch.randn(2, 3, 16, 16, generator=generator)
        targets = torch.tensor([1, 2])
        groups = ("label:1", "label:2")
    batch = Batch(tuple(f"{name}:generated-evaluation:{i}" for i in range(len(inputs))), (inputs,), targets, groups,
                  "independent-small-native-execution-not-historical-data")
    # Separate training identities; this is instrumentation validation, not a
    # held-out performance claim. Inputs are deliberately declared reused here.
    training = Batch(tuple(f"{name}:generated-training:{i}" for i in range(len(inputs))), (inputs.clone(),), targets.clone(), groups,
                     "independent-small-native-execution-training-input")
    return adapter, training, batch


def run_case(name, output, device="cpu", continuation=True):
    torch.manual_seed(2021)
    np.random.seed(2021)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    adapter, train, evaluation = make_case(name, device)
    journal = Journal(output, run_id=f"{name}-independent-portability-smoke-v1")
    start = time.perf_counter()
    capture_update(adapter, train, journal)  # initialize real optimizer memory
    donor = capture_update(adapter, train, journal)
    entry = adapter.snapshot()
    print(f"{name}: CSRG", flush=True)
    support = run_csrg(adapter, evaluation, journal)
    print(f"{name}: native causal forks", flush=True)
    horizon = (1, 2) if continuation else (1,)
    following = (train,) if continuation else ()
    branches = run_forks(adapter, train, evaluation, journal, donor=donor, horizons=horizon, continuation_batches=following)
    print(f"{name}: finite amplitude path", flush=True)
    amplitude = run_amplitude_path(adapter, evaluation, journal, realized_update=donor, horizons=horizon, continuation_batches=following)
    print(f"{name}: identity ledger", flush=True)
    ledger = run_boundary_ledger(adapter, evaluation, journal, realized_update=donor)
    restored = adapter.snapshot().identity == entry.identity
    journal.save()
    print(f"{name}: Core v3 export and independent checks", flush=True)
    core = export_core(journal.root)
    checks = verify_result(journal.root)
    result = dict(status="PASS" if restored and checks["status"] == "PASS" else "FAIL", architecture=name,
                  device=device, parameters=sum(v.numel() for v in adapter.model.parameters()),
                  samples=len(evaluation.identities), target_groups=len(support["group_order"]),
                  csrg_actual_forwards=support["actual_forward_count"], causal_branches=len(branches["branches"]),
                  horizons=horizon, finite_amplitude_states=len(amplitude["probes"]),
                  continuation_states=len(amplitude["continuations"]),
                  ledger_evaluable=ledger["adjudication"]["evaluable_count"],
                  ledger_accuracy=ledger["adjudication"]["accuracy"],
                  persistent_state_and_rng_restored=restored,
                  native_buffer_changed=branches["native_buffer_changed"],
                  core_v3=core, checks=checks, wall_seconds=time.perf_counter() - start,
                  artifact_bytes=sum(p.stat().st_size for p in journal.root.rglob("*") if p.is_file()),
                  historical_result_claim=False, input_scope="generated smoke inputs; not historical datasets",
                  instrument_source_hashes=instrument_code_hashes(),
                  versions={name: version(name) for name in ("torch", "numpy", "jsonschema")})
    (journal.root / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("architecture", "status", "wall_seconds", "artifact_bytes")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=("nanogpt", "resnet", "both"), default="both")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    names = ("nanogpt", "resnet") if args.architecture == "both" else (args.architecture,)
    for name in names:
        run_case(name, args.output / name, args.device, continuation=(name == "nanogpt"))


if __name__ == "__main__":
    main()
