# File inventory

All listed files are new in this independent repository. No upstream file was modified.

## Layout

```text
src/gfg_instruments/   scientific kernels, adapters, state/identity and verification
  adapters/            nanoGPT, ResNet, common native execution adapter
  _vendor/             exact model/Core source, licences and Core protocol
provenance/            original-source copies and fixed-commit SHA-256 manifest
tests/                 actual execution, legacy parity and negative controls
tools/                 read-only audits and one-time extraction utility
docs/                  audit, adapter contract, semantics and validation reports
reports/               lightweight machine-readable run and dependency summaries
outputs/               local raw receipts, tests and installation smoke; ignored
dist/                  built installation wheel; ignored
```

## New implementation, tests and delivery files

- `.gitattributes`
- `.gitignore`
- `AGENTS.md`
- `LICENSE`
- `README.md`
- `THIRD_PARTY_NOTICES.md`
- `docs/ADAPTER_CONTRACT.md`
- `docs/AUDIT.md`
- `docs/CONSISTENCY_REPORT.md`
- `docs/LICENSE_AUDIT.md`
- `docs/SCIENTIFIC_SEMANTICS.md`
- `docs/VALIDATION_REPORT.md`
- `provenance/UPSTREAM_FILES.json`
- `pyproject.toml`
- `reports/CHECKS.json`
- `reports/DEPENDENCY_LICENSES.json`
- `reports/INSTALLED_NANOGPT_SMOKE.json`
- `reports/NANOGPT_SMOKE.json`
- `reports/RESNET_SMOKE.json`
- `src/gfg_instruments/__init__.py`
- `src/gfg_instruments/adapters/__init__.py`
- `src/gfg_instruments/adapters/base.py`
- `src/gfg_instruments/adapters/nanogpt.py`
- `src/gfg_instruments/adapters/resnet.py`
- `src/gfg_instruments/amplitude_kernel.py`
- `src/gfg_instruments/boundaries.py`
- `src/gfg_instruments/contracts.py`
- `src/gfg_instruments/core_bridge.py`
- `src/gfg_instruments/csrg.py`
- `src/gfg_instruments/errors.py`
- `src/gfg_instruments/evidence.py`
- `src/gfg_instruments/forks.py`
- `src/gfg_instruments/identity.py`
- `src/gfg_instruments/ledger.py`
- `src/gfg_instruments/optimizers.py`
- `src/gfg_instruments/paths.py`
- `src/gfg_instruments/smoke.py`
- `src/gfg_instruments/state.py`
- `src/gfg_instruments/verify.py`
- `tests/conftest.py`
- `tests/test_adapters_optimizers.py`
- `tests/test_core_evidence.py`
- `tests/test_instruments.py`
- `tools/dependency_inventory.py`
- `tools/extract_upstream.py`
- `tools/verify_upstream.py`
- `docs/FILE_INVENTORY.md` (this file)

## Exact upstream copies

The full list of 46 upstream files, their original repository paths, commit IDs,
byte sizes and SHA-256 values is in `provenance/UPSTREAM_FILES.json`.
The wheel uses the model and Core copies under `_vendor`; historical experiment
wrappers under `provenance/upstream` are read-only comparison material.
