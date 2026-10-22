# GFG Neural Instruments

An independent extraction of four research instruments, with model operations
separated from their scientific measurement formulas. No training dataset or
historical checkpoint is required to run the small native execution checks.

This is a research package, not a claim that every architecture or optimizer has
been validated. The original frozen experiments and their published conclusions
are not modified or superseded.

## Install and run

Python 3.12+ and PyTorch are required. In a separate environment:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
gfg-instruments-smoke --output outputs/my-first-run
gfg-instruments-verify outputs/my-first-run/nanogpt
gfg-instruments-verify outputs/my-first-run/resnet
```

Use a fresh output directory: evidence overwrites are rejected. The smoke uses
CPU by default, a tiny **real nanoGPT** and the actual **CIFAR ResNet-18** model.
It exercises all four instruments and the unmodified GFG Core v3 validator.
Generated smoke inputs are not historical scientific data; the printed ledger
accuracy is only a plumbing diagnostic, not a performance reproduction.

The complete state journal is intentionally materialized. ResNet measurements
can occupy gigabytes, even though the inputs are small. No model downloads,
network calls, original repository checkout, or private files are needed at run
time. See [validation](docs/VALIDATION_REPORT.md) for measured storage and results.

## Four instruments

| API | Model-independent measurement | Adapter supplies |
| --- | --- | --- |
| `csrg.run_csrg` | q10, necessity, pair backup, single/double failure slack, allocation, concentration, effective support | Four registered intervention sites; actual gated forwards; explicit targets |
| `forks.run_forks` | Native/skip/parameter/memory/donor forks, matched continuation, full response differences and interaction residuals | Native optimizer execution; exact restorable parameters, buffers, memory and RNG |
| `paths.run_amplitude_path` | Original 11-scale, five-center finite-amplitude kernel, J/K and Simpson diagnostics, categorical transitions and continuation | Actual captured update; real forward and continuation execution |
| `ledger.run_boundary_ledger` | Sealed identity-aligned boundary prediction, four transitions, group capability changes and adjudication | Stable target identities, frozen boundary contract and native endpoints |

The identity ledger retains the historical `m + J + K/2` finite-difference
candidate. It is **not** TL-P01's `quadratic_complete` predictor and makes no
claim to reproduce its 91.43% accuracy. A separate `seal_prediction` API accepts
other explicitly named, provenance-bound predictors without replacing formulas.

CSRG remains a four-component protocol: two baseline repetitions, four single
gates and six pair gates. It is not the 16-mask Shapley experiment. It does not
assert that a new architecture has exactly four natural components.

## Minimal usage

```python
from gfg_instruments.evidence import Journal
from gfg_instruments.forks import capture_update, run_forks
from gfg_instruments.csrg import run_csrg
from gfg_instruments.paths import run_amplitude_path
from gfg_instruments.ledger import run_boundary_ledger
from gfg_instruments.core_bridge import export_core
from gfg_instruments.verify import verify_result

# adapter = your NanoGPTAdapter, ResNetAdapter, or TorchAdapter subclass.
# train_batch / evaluation_batch = explicit Batch contracts, including IDs.
journal = Journal('outputs/a-new-run')
actual_update = capture_update(adapter, train_batch, journal)
adapter.restore(actual_update[0].pre)
support = run_csrg(adapter, evaluation_batch, journal)
forks = run_forks(adapter, train_batch, evaluation_batch, journal,
                  donor=actual_update)
path = run_amplitude_path(adapter, evaluation_batch, journal,
                          realized_update=actual_update)
ledger = run_boundary_ledger(adapter, evaluation_batch, journal,
                             realized_update=actual_update)
journal.save()
export_core(journal.root)
print(verify_result(journal.root))
```

Memory transplantation requires initialized optimizer memory. Warm up through a
real update first if starting from an uninitialized model. Undefined memory is
not silently invented. The measurement functions restore the receiving state;
`capture_update` intentionally performs and leaves the actual training update.

## Adapting a model

Subclass `TorchAdapter`, register four component identities and implement the
actual gated forward, training forward/loss, and target readout. State, optimizer,
evidence and measurement code are shared. See [adapter contract](docs/ADAPTER_CONTRACT.md),
the two reference adapters, and the independent third-network adapter in
`tests/test_adapters_optimizers.py`.

Task contracts currently support classification margins and frozen residual
candidate margins. A fundamentally different boundary family needs a versioned
boundary extension and tests, not an adapter-supplied replacement metric.
Adam, AdamW and SGD Momentum have separate checked memory semantics; unsupported
optimizers fail explicitly.

## Evidence and scientific scope

- [Code audit and original call chains](docs/AUDIT.md)
- [Preserved formulas and explicit versioned corrections](docs/SCIENTIFIC_SEMANTICS.md)
- [Original/new implementation comparisons](docs/CONSISTENCY_REPORT.md)
- [Actual tests, negative controls and limitations](docs/VALIDATION_REPORT.md)
- [Licence audit](docs/LICENSE_AUDIT.md)
- [Exact upstream files and SHA-256](provenance/UPSTREAM_FILES.json)

The journal retains native outputs, parameter/buffer states, optimizer moments,
gradients, identities and ordered receipts. Core v3 records exact input bindings,
occurrences, generated origins and explicit dispositions. Independent checkers
recompute scientific quantities from native output payloads. Hash integrity and
Core validity do not themselves establish the truth of a scientific theory.

Complete original research evidence:
[frozen research repository](https://github.com/wind342/gfg-training-learning-inference-experiments/tree/36dab5ce347dbbdac157ef23205f556606d18294),
tag `paper-experiments-cross-system-feedback-release`.
Structural Core authority:
[GFG Core structural experiments](https://github.com/wind342/gfg-core-structural-experiments/tree/6ee8a33f0268e07304d83cc3e9d87411c17a23e9).

## License

Apache-2.0 for this scoped extraction and new code; the embedded nanoGPT model
remains MIT. See [third-party notices](THIRD_PARTY_NOTICES.md). The original
evidence repository's licensing and all frozen identities remain unchanged.
No private datasets, historical tensor archives or checkpoints are redistributed.
