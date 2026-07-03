# Known Issues

This page tracks confirmed correctness bugs found during codebase review that
have not yet been fixed.

## `persistent_homology.py` — `LocalPD` and `get_local_persistence`

Quarantined: both raise `NotImplementedError` on use rather than silently
hitting the underlying `AttributeError`. They were written against
`neighborhood.get_persistence_diagram(...)` as a bound method on
GlassAtoms/ASE `Atoms`, but that functionality was moved out into the
standalone `get_persistence_diagram(atoms, ...)` function in the same module,
and the call sites were never updated. `get_local_persistence` additionally
calls `neighborhood.center()`, which does ASE's vacuum-padding centering, not
the minimum-image centering it needs. The standalone `get_persistence_diagram`
function itself is unaffected and works correctly on its own. Fixing this
needs the call sites rewritten to use the standalone function and proper
minimum-image centering (`LocalPD.center_atoms` already implements the
latter but isn't used by `get_local_persistence`).
