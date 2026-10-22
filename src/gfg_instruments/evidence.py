"""Small lossless artifact journal. Tensor payloads deduplicate, occurrences do not."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import torch

from .errors import require
from .identity import canonical, digest


class Journal:
    def __init__(self, root, run_id=None):
        self.root = Path(root)
        require(not self.root.exists(), "EVIDENCE_OUTPUT_ALREADY_EXISTS")
        self.root.mkdir(parents=True)
        (self.root / "objects").mkdir()
        self.run_id = run_id or str(uuid.uuid4())
        self.sources = {}
        self.events = []
        self._known = {}
        self._stored = set()
        self._sources_by_content = {}

    def _bytes(self, data, suffix):
        sha = hashlib.sha256(data).hexdigest()
        relative = f"objects/{sha}{suffix}"
        if relative in self._stored:
            return {"path": relative, "sha256": sha, "size": len(data)}
        path = self.root / relative
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(data)
        else:
            require(path.read_bytes() == data, "ARTIFACT_HASH_COLLISION_OR_CORRUPTION")
        self._stored.add(relative)
        return {"path": relative, "sha256": sha, "size": len(data)}

    def encode(self, value):
        if is_dataclass(value):
            return self.encode(value.__dict__)
        if isinstance(value, torch.Tensor):
            data = value.detach().cpu().contiguous()
            raw = data.reshape(-1).view(torch.uint8).numpy().tobytes()
            return {"tensor": "torch", "dtype": str(data.dtype), "shape": list(data.shape),
                    **self._bytes(raw, ".bin")}
        if isinstance(value, np.ndarray):
            data = np.ascontiguousarray(value)
            require(not data.dtype.hasobject, "OBJECT_ARRAY_NOT_SERIALIZABLE")
            return {"tensor": "numpy", "dtype": str(data.dtype), "shape": list(data.shape),
                    **self._bytes(data.tobytes(), ".bin")}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): self.encode(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [self.encode(v) for v in value]
        return value

    def artifact(self, payload):
        return self._bytes(canonical(self.encode(payload)), ".json")

    def source(self, identity, payload):
        signature = (identity, digest(payload))
        if signature in self._sources_by_content:
            return self._sources_by_content[signature]
        artifact = self.artifact(payload)
        key = "source:" + digest({"identity": identity, "artifact": artifact})
        row = dict(id=key, identity=identity, artifact=artifact)
        self.sources[key] = row
        self._known[key] = "source"
        self._sources_by_content[signature] = key
        return key

    def record(self, operation, parents, payload, disposition=None):
        require(parents and len(set(parents)) == len(parents), "MISSING_OR_DUPLICATE_EXPLICIT_INPUT_BINDING")
        require(all(p in self._known for p in parents), "UNKNOWN_INPUT_BINDING")
        artifact = self.artifact(payload)
        index = len(self.events)
        key = f"{self.run_id}:occurrence:{index}"
        material = dict(id=key, index=index, operation=operation, parents=list(parents),
                        artifact=artifact, disposition=disposition,
                        prior=self.events[-1]["receipt_sha256"] if self.events else None)
        row = {**material, "receipt_sha256": digest(material)}
        self.events.append(row)
        self._known[key] = "generated"
        return key

    def save(self):
        document = {"schema": "neural-instrument-journal-v1", "run_id": self.run_id,
                    "sources": list(self.sources.values()), "events": self.events}
        document["journal_sha256"] = digest(document)
        path = self.root / "journal.json"
        require(not path.exists(), "JOURNAL_ALREADY_SEALED")
        path.write_bytes(canonical(document))
        return document


def load_artifact(root, reference):
    root = Path(root).resolve()
    path = (root / reference["path"]).resolve()
    require(path.is_relative_to(root), "ARTIFACT_PATH_ESCAPE")
    raw = path.read_bytes()
    require(len(raw) == reference["size"] and hashlib.sha256(raw).hexdigest() == reference["sha256"], "ARTIFACT_HASH_MISMATCH")
    return raw


def decode(root, value):
    if isinstance(value, dict) and "tensor" in value:
        raw = load_artifact(root, value)
        if value["tensor"] == "numpy":
            return np.frombuffer(raw, dtype=value["dtype"]).copy().reshape(value["shape"])
        dtype = getattr(torch, value["dtype"].removeprefix("torch."))
        data = torch.frombuffer(bytearray(raw), dtype=dtype) if raw else torch.empty(0, dtype=dtype)
        return data.clone().reshape(value["shape"])
    if isinstance(value, dict):
        return {k: decode(root, v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(root, v) for v in value]
    return value


def payload(root, reference):
    return decode(root, json.loads(load_artifact(root, reference)))


def validate_journal(root):
    root = Path(root)
    document = json.loads((root / "journal.json").read_text("utf-8"))
    require(digest({k: v for k, v in document.items() if k != "journal_sha256"}) == document["journal_sha256"], "JOURNAL_HASH_MISMATCH")
    known, prior = set(), None
    checked = {}
    def verify_object(ref):
        if ref["path"] in checked:
            require(checked[ref["path"]] == ref, "CONFLICTING_ARTIFACT_REFERENCES")
            return
        raw = load_artifact(root, ref)
        checked[ref["path"]] = ref
        if ref["path"].endswith(".json"):
            def walk(value):
                if isinstance(value, dict):
                    if "tensor" in value:
                        verify_object({k: value[k] for k in ("path", "sha256", "size")})
                    else:
                        for child in value.values():
                            walk(child)
                elif isinstance(value, list):
                    for child in value:
                        walk(child)
            walk(json.loads(raw))
    for source in document["sources"]:
        require(source["id"] not in known, "DUPLICATE_SOURCE")
        require(source["id"] == "source:" + digest({"identity": source["identity"], "artifact": source["artifact"]}), "SOURCE_IDENTITY_MISMATCH")
        verify_object(source["artifact"])
        known.add(source["id"])
    for index, row in enumerate(document["events"]):
        require(row["index"] == index and row["id"] == f"{document['run_id']}:occurrence:{index}", "OCCURRENCE_IDENTITY_MISMATCH")
        require(row["id"] not in known and row["parents"] and len(set(row["parents"])) == len(row["parents"])
                and set(row["parents"]) <= known, "INVALID_GENERATION_BINDING")
        require(row["prior"] == prior and digest({k: v for k, v in row.items() if k != "receipt_sha256"}) == row["receipt_sha256"], "RECEIPT_CHAIN_MISMATCH")
        verify_object(row["artifact"])
        known.add(row["id"])
        prior = row["receipt_sha256"]
    return document
