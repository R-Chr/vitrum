"""BALACE — batch active learning for ACE potentials. Experimental and unsupported.

This module drives VASP/LAMMPS active-learning workflows through FireWorks. It needs
external services the test suite cannot reach, has no coverage, and has confirmed bugs
documented in `docs/vitrum/known_issues.md` — `run_train_pace()` raises `TypeError`,
`train_pace` submits its workflow twice, and several module-level functions still take
`self`. It is shipped as-is, is not covered by the package's semantic-versioning promise,
and may be split into a separate distribution or removed in a future release.
"""

import warnings

warnings.warn(
    "vitrum.batch_active (BALACE) is experimental and unsupported: it has known correctness "
    "bugs and no test coverage. See docs/vitrum/known_issues.md. It is excluded from vitrum's "
    "semantic-versioning promise and may be split out or removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
