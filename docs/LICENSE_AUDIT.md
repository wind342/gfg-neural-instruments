# Scoped pre-extraction licence audit

The repository owner explicitly requested a new open-source instrument extraction
on 2026-10-22. This grant covers the owner's instrument code extracted here, not a
relicensing of the original research-evidence repository or its scientific data.

| Material | Authority | Treatment |
| --- | --- | --- |
| Owner-authored instrument code and ResNet reference | Frozen evidence commit `36dab5ce347dbbdac157ef23205f556606d18294`; explicit owner authorization | Scoped Apache-2.0 extraction; source identities retained |
| GFG Core v3 and compatibility projections | `gfg-core-structural-experiments`, Apache-2.0 | Exact source copies, protocol and licence retained; no Core formula/validator changes |
| nanoGPT model | Karpathy nanoGPT commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`, MIT verified through GitHub's licence endpoint | Exact model.py and complete MIT notice retained |
| PyTorch | Installed distribution's licence and bundled third-party notices | External dependency, no wheel or binaries redistributed |
| NumPy | BSD-3-Clause and distribution notices | External dependency |
| jsonschema | MIT | External dependency |
| pytest | MIT | External test dependency |
| setuptools/wheel | MIT | External build dependencies |

No datasets, checkpoints, private journals, paper PDFs, credentials, historical
tensor bundles, or downloaded binary packages will be copied. Runtime smoke
inputs are independently generated and explicitly not historical experiment data.
No network upload is authorized. Dependency versions and available installed
licence metadata are recorded by the validation runner.

The original evidence repository's PROPRIETARY_NOTICE remains unchanged. This
audit does not grant any rights in third-party data or code beyond their notices.

## Installed dependency check

`reports/DEPENDENCY_LICENSES.json` records 22 direct/transitive runtime, test and
build distributions, their versions, active dependencies, declared expressions
and available licence-file SHA-256 values. These are read from installed metadata,
not inferred from package popularity. Null metadata is left null; the script
does not invent a licence label. The Colorama, Jinja2 and setuptools licence files
were also read when their metadata field was empty (BSD-style, BSD-style and MIT
permission text respectively).

Observed expressions include MIT, BSD variants, Apache-2.0 OR BSD-2-Clause,
PSF-2.0, and NumPy's bundled permissive-license expression. No dependency binary
is copied into the wheel. No conflicting redistribution condition was found for
the specific source copies shipped here; bundled CUDA and other external binary
licences are not being relicensed or redistributed by this project.
