# Validation report — 2026-10-22

## Actual completed checks

- Unit/regression suite: **35 passed, 0 failed, 0 errors, 0 skipped**, 15.431 s.
- Upstream source manifest: **46 files** passed SHA-256 and length checks.
  **44** were additionally compared byte-for-byte with the locally available
  fixed Git commits; two nanoGPT files were obtained from the fixed upstream
  GitHub commit and verified against their recorded extraction hashes.
- Original evidence and structural Core worktrees remain clean. The evidence
  release tag still resolves to `36dab5ce347dbbdac157ef23205f556606d18294`.
- Actual package build, isolated target install and outside-checkout execution
  passed. The imported package path was `outputs/installed-package`, not `src`
  and not either original repository.

## Four-instrument native runs

| Run | Parameters | CSRG measurements checked | Native updates checked | Core occurrences / relations | Total wall | Artifacts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nanoGPT, source checkout | 6,416 | 31 | 15 | 523 / 1,759 | 52.32 s | 18,076,621 B |
| CIFAR ResNet-18, source checkout | 11,171,397 | 19 | 3 | 312 / 1,063 | 154.49 s | 1,216,392,245 B |
| nanoGPT, installed wheel | 6,416 | 31 | 15 | 523 / 1,759 | 52.41 s | 18,076,481 B |

All three returned PASS for artifact/receipt hashes, numerical recomputation and
unmodified Core v3 validation. Persistent state and RNG restoration passed.
ResNet's real BN changes were detected separately from its parameter deltas.
Times include journal materialization and Core verification; concurrent developer
checks occurred, so these are execution records, **not throughput benchmarks**.

Each CSRG measurement performed 12 actual forwards. Each architecture ran seven
causal branches, the 11-state amplitude path and the identity boundary ledger.
nanoGPT also ran horizon 2 continuation across all seven branches and all five
amplitude centers. ResNet used horizon 1 only. Inputs: 12 generated nanoGPT
examples, 2 generated 16x16 ResNet images. These are real native computations but
not held-out scientific-performance experiments.

Machine-readable records: `reports/NANOGPT_SMOKE.json`, `RESNET_SMOKE.json`,
`INSTALLED_NANOGPT_SMOKE.json`. Complete tensor receipts stay under `outputs/`
and are excluded from Git. The two final source-run journals total about 1.15 GiB;
the installed nanoGPT check adds about 17.2 MiB. Development trials are also kept,
not silently deleted or represented as final results.

## Independent installation

Built `gfg_neural_instruments-0.1.0-py3-none-any.whl`, **94,923 bytes**, SHA-256
`50c6d92bd4eac3475edc53913bcf92c967880f748886f15f034019ef598b842a`.
Archive inspection found 49 files, both upstream licences, required Core protocol
JSON, and no `__pycache__`/`.pyc`. Installed with `--no-deps --target` into a new
directory using the existing dependency environment; no shared Python environment
was modified. From a separate working directory, the installed nanoGPT smoke
ran all four instruments and Core verification successfully.

This is an isolated-package-install test, not a fresh-OS installation of every
dependency. Tested dependency versions: PyTorch 2.11.0+cu128, NumPy 2.5.2,
jsonschema 4.25.1, pytest 9.0.2. Runs used deterministic CPU execution, two Torch
threads. The broader declared dependency ranges have not all been tested.

## Negative controls

Tests reject: duplicate target IDs; changed content under the same ID; invalid
gates; branch storage aliasing; nonfinite state/memory; incompatible optimizer
settings; unsupported optimizer memory; altered native-update receipts;
tampered raw-output margins; altered prediction seals; endpoint read before seal;
missing generation inputs; missing Core primary evidence; orphan source coverage;
corrupted artifact bytes. Zero/unknown predictions and uninformative signatures
remain explicitly not evaluable.

## Development failures retained

1. First pytest invocation lacked the parent of its temporary output directory.
   It had 15 passes and 14 fixture setup errors. Creating the new output parent
   fixed that setup issue; subsequent full suites passed.
2. During a development end-to-end run the Core bridge source was edited while
   the process was running. Core rejected the resulting predicate/source identity
   mismatch (`PREDICATE_PROFILE_UNKNOWN:SNAPSHOT_REGISTRY_MISMATCH`). That run is
   **not** reported as PASS. Final runs used unchanged runtime source throughout
   and all passed. Do not edit runtime code during a captured run.

## Limits

Historical whole-experiment byte replay: **NOT RUN**. Large historical payloads
were not available/bundled. This release does not reproduce TL-P01 accuracy,
rerun long training, or replace old scientific conclusions. No full U-Net
adapter, GPU parity, AMP/distributed optimizer validation or large-scale storage
performance claim is included. See `CONSISTENCY_REPORT.md` for exact comparison
scope and `SCIENTIFIC_SEMANTICS.md` for explicit corrections.

No remote repository was created, pushed, or published. This is a local,
independent Git repository; original repositories and frozen evidence remain
unchanged.

One nonfunctional Git whitespace warning remains: the extracted amplitude kernel
retains an extra blank line at EOF from the copied function block. It is not a
test or runtime failure; the tested source bytes are deliberately left unchanged.
