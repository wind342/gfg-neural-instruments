# Original/new implementation consistency

All original sources refer to the immutable commits in `UPSTREAM_FILES.json`.
The tests execute original function bodies or exact original arithmetic blocks
against the new implementations, using independently generated inputs.

| Comparison | Check |
| --- | --- |
| Original `builder.decision_outputs` vs new readout | Exactly equal margins, argmax and group q10 for the original 212x24 shape; smaller sizes exercised separately |
| Original stepwise support arithmetic block vs new CSRG | Exact necessity, pair backup, slacks and effective support; allocation/concentration match on their defined domain |
| Full measurement scope | 12 native forward results, seven support arrays; 67 response roles, not only baseline logits |
| Original `_numeric_path` | Python AST equality for the complete copied function; independent recomputation of actual native path J/K, Simpson, endpoints and categorical transitions |
| Original reciprocal signature | Exact original normalized RMS/median function parity; uninformative case stays None |
| Original reciprocal classifier | Fully Boolean cases retain original decisions; all-None case demonstrates the old false classification and the new NOT_EVALUABLE result |
| Native optimizer / restore | Real Adam, AdamW, SGD Momentum, dampening and Nesterov; restored native steps repeat exactly; transported innovations agree numerically (FP32 tolerance stated in tests) |
| Model implementation | nanoGPT and CIFAR ResNet source SHA-256 identical to fixed upstream bytes; gate outputs compared with native models |
| Parameters vs BN buffers | Real ResNet update changes BN; parameter-only intervention retains receiver buffers; native full state preserves actual BN changes |
| Identity ledger | Actual endpoint follows sealed prediction; identity permutation aligns, changed content and pre-seal endpoint are rejected; group means/directions recomputed |
| Core v3 | Actual unmodified Core build/validate plus artifact hashes, exact dependency bindings and independent native-output numerical checks |

## Explicit nonclaims

- Full historical checkpoints, training datasets and tensor archives were not
  available/bundled. Historical end-to-end byte replay was **not run**.
- Small generated real executions validate the instruments, not the original
  paper's accuracy, conclusions or long training dynamics.
- The standalone third-network adapter is a portability test, not new scientific
  evidence about its architecture.
- DDPM data-ID coupling was audited; a diffusion U-Net adapter is **not shipped**.
  Residual-candidate readout is tested independently.
- Full reciprocal horizons to 100 and historical continuation to 500 were not
  executed in this smoke. APIs require supplied matched batches for such runs.
- Cross-device byte identity, distributed execution and throughput at large-model
  scale are unverified. CPU deterministic execution is the tested profile.

Actual run totals, software versions and test status are in
`VALIDATION_REPORT.md` and the lightweight `reports/` records. Undefined-domain
corrections are documented separately; no historical result was rewritten.
