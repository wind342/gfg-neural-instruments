# Pre-extraction code audit

Authority: frozen evidence commit `36dab5ce347dbbdac157ef23205f556606d18294`.
This audit read executable code and protocol/validator call chains, not just README.
The exact consulted sources are retained in `provenance/upstream` with hashes.

| Instrument | Original call chain and coupling | Extraction decision |
| --- | --- | --- |
| CSRG-4C | support_redundancy `builder.decision_outputs`; stepwise `runtime.support_probe`; component registry/contracts; archive verifier | Preserve q10, singles/pairs, clipping, loads and 12 forwards; replace 212x24 assertions, groups 0..22, four hardcoded transformer module names with explicit contracts |
| Causal forks | stepwise native runtime -> `branches.py`/`reciprocal.py` -> `reciprocal_validator.py` | Preserve actual update, skip, parameter/memory isolation, donor innovation, matched continuation and response differences; detach historical state-manifest paths and Adam-specific mapping |
| Finite amplitude | stepwise `amplitude_path.py::_numeric_path` -> amplitude validator and native-response validator | Copy numeric kernel verbatim; keep 11 scales, five centers, epsilon .125, J/K/Simpson, full forward outputs and categorical transitions; use adapter state operations |
| Identity ledger | identity_aligned_margin_crossing `build_and_seal_pretarget_projection` -> `adjudicate_after_seal` -> `validate_completed_experiment` | Preserve pretarget seal, identity alignment, margin J/K candidate and capability/direction calculation; remove dependencies on historical V2 index and absolute archive layouts |

Original source families are named in full in the upstream manifest. Production
runtime never imports these historical wrappers or requires their external data.

## Specific issues checked

- `reciprocal_validator._signature_row` can return `aggregate_donor_closer=None`.
  `_classification` wraps this in `bool`, allowing an uninformative record to
  contribute to RECEIVING_STATE_DOMINANT. The new classifier preserves unknown.
- The ResNet analysis clones the full `state_dict`, forms float deltas and adds
  them, including BN running statistics. Therefore a historical path can be a
  combined parameter/buffer intervention. This is not silently described as a
  pure parameter direction in the new package.
- DDPM fixed evaluation uses local positional IDs; shuffled training also uses
  positional ranges. Positions alone do not establish dataset sample identity.
  New inputs require IDs bound to actual content; no historical claim is changed.
- Lightweight `CompactGFG.validate` checks a narrower set of references than
  full Core v3. The new bridge calls the actual unchanged Core validator and
  separately verifies artifact payloads, native outputs and numerical results.
- Historical amplitude `_rms` returns zero if no finite elements exist. New
  diagnostic `finite_rms` returns None; no-observation is not zero response.
- Historical zero-total necessity uses zero allocation/concentration but NaN
  effective support. New output uses explicit undefined allocation/concentration
  as well; the change is versioned and limited to that undefined domain.

## What remains unchanged

Original repositories, source files, experiment results, verifiers, protocols,
frozen tags and commits were not edited. No old archive is rehashed or regenerated.
Core and the two upstream model implementations are byte-identical copies.
Historical dataset/checkpoint files are not needed, copied or redistributed.

## Dependence boundary

The original full experiment runners depend on many archive-specific utilities;
those are **not** packaged as a runnable legacy environment. Only extracted
instrument formulas and the complete minimal Core runtime/protocol dependencies
are used in the new package. The original source copies are comparison material,
not an import fallback to the original repository.
