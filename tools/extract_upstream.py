"""One-time, fail-closed, hash-recorded extraction; never writes upstream."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FROZEN = "36dab5ce347dbbdac157ef23205f556606d18294"
CORE_COMMIT = "6ee8a33f0268e07304d83cc3e9d87411c17a23e9"
NANOGPT = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    args = parser.parse_args()
    rows = []

    def put(data, relative, repository, commit, source):
        path = ROOT / relative
        if path.exists():
            raise RuntimeError(f"REFUSE_OVERWRITE:{path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        rows.append(dict(path=relative, repository=repository, commit=commit,
                         source=source, sha256=hashlib.sha256(data).hexdigest(),
                         size=len(data)))

    def gitcopy(repo, commit, source, dest, name):
        data = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{source}"])
        put(data, dest, name, commit, source)

    core_files = subprocess.check_output(
        ["git", "-C", str(args.core), "ls-tree", "-r", "--name-only", CORE_COMMIT], text=True
    ).splitlines()
    for source in core_files:
        if source.startswith(("src/generation_relation_core/", "compat/", "protocol/core_v3/")) or source == "LICENSE":
            gitcopy(args.core, CORE_COMMIT, source,
                    "src/gfg_instruments/_vendor/core/" + source,
                    "wind342/gfg-core-structural-experiments")
    families = {
        "gfg_nanogpt_support_redundancy_v1": ["builder.py", "runtime.py", "capture_contract_v2.json", "support_gfg.py", "verify_archive.py"],
        "gfg_nanogpt_stepwise_support_transition_v1": ["runtime.py", "contracts.py", "branches.py", "reciprocal.py", "reciprocal_validator.py", "RECIPROCAL_MATCHED_PAIR_PROTOCOL_V2.json", "amplitude_path.py", "amplitude_path_validator.py", "B_UPDATE_AMPLITUDE_PATH_PROTOCOL.json", "native_response_500.py", "native_response_500_validator.py"],
        "gfg_nanogpt_identity_aligned_margin_crossing_v1": ["experiment.py"],
        "gfg_resnet_cifar_training_learning_generalization_v1": ["model.py", "analysis.py", "PROTOCOL_FREEZE.md"],
        "gfg_ddpm_cifar_training_learning_generalization_v1": ["data.py", "runner.py", "boundary.py", "PROTOCOL_FREEZE.md"],
    }
    for directory, files in families.items():
        for name in files:
            source = f"experiments/{directory}/{name}"
            gitcopy(args.evidence, FROZEN, source, "provenance/upstream/" + source,
                    "wind342/gfg-training-learning-inference-experiments")
    source = "experiments/gfg_resnet_cifar_training_learning_generalization_v1/model.py"
    gitcopy(args.evidence, FROZEN, source, "src/gfg_instruments/_vendor/resnet_model.py",
            "wind342/gfg-training-learning-inference-experiments")
    for name in ("model.py", "LICENSE"):
        data = subprocess.check_output(["gh", "api", f"repos/karpathy/nanoGPT/contents/{name}?ref={NANOGPT}",
                                        "-H", "Accept: application/vnd.github.raw+json"])
        put(data, "src/gfg_instruments/_vendor/nanogpt/" + name, "karpathy/nanoGPT", NANOGPT, name)
    manifest = dict(schema="gfg-neural-instruments-upstream-v1", files=rows)
    (ROOT / "provenance/UPSTREAM_FILES.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"copied_files": len(rows), "bytes": sum(r["size"] for r in rows)}))


if __name__ == "__main__":
    main()
