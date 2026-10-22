"""Read-only verification of the extraction manifest and optional Git authorities."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--core", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "provenance/UPSTREAM_FILES.json").read_text("utf-8"))
    repos = {"wind342/gfg-training-learning-inference-experiments": args.evidence,
             "wind342/gfg-core-structural-experiments": args.core}
    matched = 0
    for row in manifest["files"]:
        path = (ROOT / row["path"]).resolve()
        assert path.is_relative_to(ROOT), "UPSTREAM_PATH_ESCAPE"
        raw = path.read_bytes()
        assert len(raw) == row["size"] and hashlib.sha256(raw).hexdigest() == row["sha256"], row["path"]
        if repos.get(row["repository"]):
            original = subprocess.check_output(["git", "-C", str(repos[row["repository"]]), "show", f"{row['commit']}:{row['source']}"])
            assert original == raw, "ORIGINAL_GIT_BYTE_MISMATCH:" + row["path"]
            matched += 1
    upstream_status = {}
    for name, path in repos.items():
        if path:
            upstream_status[name] = {
                "head": subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip(),
                "status_porcelain": subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()}
    print(json.dumps({"status": "PASS", "files": len(manifest["files"]),
                      "git_authority_byte_matches": matched, "upstream_status": upstream_status}, indent=2))


if __name__ == "__main__":
    main()
