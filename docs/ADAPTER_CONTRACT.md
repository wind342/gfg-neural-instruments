# Model/task/optimizer contract

The minimal public protocol is `contracts.ModelAdapter`. The reference base
`adapters.TorchAdapter` implements state capture/restoration and the actual
optimizer step. An adapter must supply:

1. Four `Component(identity, intervention_site, parameter_names)` records. IDs
   and intervention locations are frozen before measurement. A gate must act at
   the declared native location; replacing the network with a fitted surrogate
   does not satisfy this contract.
2. `_forward(batch, gates)` returning real functional readouts. The public
   `forward` runs eval/no-grad and restores all module training modes. CSRG checks
   that parameters, buffers, optimizer memory, gradients and RNG are unchanged.
3. `training_forward(batch)` returning the real output and scalar training loss.
   The shared implementation performs backward, declared clipping, exactly one
   native optimizer step, and captures actual before/after tensors.
4. `evaluation_targets(batch)` if targets differ from `batch.targets`, e.g.
   nanoGPT's last-position labels. Measurement formulas are not supplied here.
5. A frozen `BoundaryContract` and `Batch` with explicit sample IDs, group IDs,
   source authority, actual inputs/targets and any fixed candidate tensors.

## State and identity

Snapshot includes parameters **and buffers**, optimizer state and name-aligned
parameter groups, submodule modes, saved gradients, Python/NumPy/Torch/CUDA RNG.
State restore rejects incompatible names/shapes/dtypes, nonfinite values and
incompatible optimizer layouts. Journal deserialization uses explicit tensors
and JSON, not executable pickle. `Snapshot.from_record` rehydrates saved states.

Branch snapshots must have disjoint tensor storage. A native full step includes
BN changes; parameter-only and memory-only initial branches hold receiver BN
buffers. Later training follows the declared train/eval contract and may update
buffers normally. Buffers never silently enter the parameter delta.

Batch identity includes each stable ID's content hash. Reordering is supported;
reusing an ID for different content fails ledger alignment. The library cannot
infer a dataset's true identity from an anonymous tensor: adapters must preserve
upstream sample identity. Tests reject duplicate IDs and position-based
misalignment, but do not claim to repair historical DDPM archives retroactively.

## Optimizers

- Adam/AdamW: separate accepted optimizer types, native update execution;
  historical memory innovation `post - beta * pre`, injected into receiver
  beta-decayed memory, and exact native step-count increment.
- SGD Momentum: native momentum innovation `post - momentum * pre`; no Adam
  second-moment formula. Dampening and Nesterov are tested against native steps.
- Equal optimizer settings and parameter identities are required for transport.
  Uninitialized memory, unknown layouts, RMSprop and AMSGrad are rejected.
  Fused, distributed, mixed-precision/scaler and differentiable optimizers are
  not validated by this release.

## Boundaries

Classification uses target minus maximum competitor logit, with an explicit
argmax/positive/nonnegative correctness rule. Residual candidate tasks use the
minimum candidate-minus-truth squared-error gap, under a frozen candidate bank.
The CSRG group quantile remains linear-interpolated q10. New boundary families
require a named/versioned scientific contract; an adapter cannot silently change
these formulas.

## Scope of defaults

Fork/path smoke defaults to horizon 1. Longer measured horizons require exactly
the corresponding continuation batches and matched RNG opportunities.
`adjudicate_reciprocal` requires genuinely reciprocal donor/receiver identities;
the frozen classification requires horizons 1, 2, 5, 20, 100 in both directions.
Short runs explicitly return NOT_EVALUABLE for that classification.

The third-party-style adapter in the tests is outside the package and runs the
same CSRG without edits to the instrument core. Production GPU throughput,
distributed models and recurrent hidden-state adapters are not validated here.
