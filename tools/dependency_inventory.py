"""Inspect installed dependency metadata and licence-file hashes; no install."""
import hashlib
from importlib.metadata import distribution
import json

from packaging.requirements import Requirement


def main():
    pending = ["torch", "numpy", "jsonschema", "pytest", "setuptools"]
    records = {}
    while pending:
        name = pending.pop()
        normalized = name.lower().replace("_", "-")
        if normalized in records:
            continue
        dist = distribution(name)
        licenses = []
        for path in dist.files or []:
            # Distribution-owned notices only; do not copy dependency binaries.
            if ".dist-info/" in str(path) and str(path).lower().endswith(("license", "license.txt", "license.md", "copying")):
                raw = dist.locate_file(path).read_bytes()
                licenses.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        dependencies = []
        for spec in dist.requires or []:
            req = Requirement(spec)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                dependencies.append(req.name)
                pending.append(req.name)
        records[normalized] = dict(version=dist.version, license_expression=dist.metadata.get("License-Expression"),
                                   license=dist.metadata.get("License", "")[:1000], license_files=licenses,
                                   active_dependencies=dependencies)
    print(json.dumps({"schema": "installed-dependency-licence-inventory-v1", "redistributes_binaries": False,
                      "dependencies": records}, indent=2))


if __name__ == "__main__":
    main()
