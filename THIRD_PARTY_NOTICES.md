# Third-party and upstream notices

This package contains:

1. GFG Core v3, compatibility projections and protocol files from
   `wind342/gfg-core-structural-experiments`, commit
   `6ee8a33f0268e07304d83cc3e9d87411c17a23e9`, Apache-2.0.
   Full licence: `src/gfg_instruments/_vendor/core/LICENSE`.
2. nanoGPT `model.py`, Copyright (c) 2022 Andrej Karpathy, commit
   `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`, MIT.
   Full notice: `src/gfg_instruments/_vendor/nanogpt/LICENSE`.
3. Owner-authorized instrument source/protocol extracts and CIFAR ResNet model
   from `wind342/gfg-training-learning-inference-experiments`, frozen commit
   `36dab5ce347dbbdac157ef23205f556606d18294`. Licensed for this extraction under
   Apache-2.0 by the owner; this does not relicense the original repository.

Unmodified upstream copies are identified individually in
`provenance/UPSTREAM_FILES.json`. Modified/extracted formulas are documented in
`docs/SCIENTIFIC_SEMANTICS.md`; full upstream source copies remain available for
comparison. No third-party licence notice has been removed.

PyTorch, NumPy, jsonschema, pytest, setuptools and wheel are external dependencies;
their wheels, CUDA libraries and bundled third-party code are not redistributed
in this repository. Their own distribution licences continue to apply.
