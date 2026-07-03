# Known Issues

This page tracks confirmed correctness bugs found during a codebase review.
They are documented here rather than fixed so they can be prioritized and
addressed deliberately; none of the code below has been changed to work
around them.

## `scattering.py` — `Scattering.get_N_running`

`self.chemical_symbols` is a plain Python `list`, not a `numpy.ndarray`.
Comparing a list to a string (`self.chemical_symbols == pair[0]`) evaluates
to a scalar `False` rather than an elementwise boolean array, so
`len(np.where(...))` is always exactly `1`, regardless of species. This means
the number density used in the running-coordination-number integral is wrong
for every pair. Fix: convert to `np.array(self.chemical_symbols)` before
comparison (or store it as an array in `__init__`).

## `trajectory.py` — `get_high_low_displacement_index`

Same bug pattern as above: `initial_state.get_chemical_symbols() == target_atom`
compares a list to a string and does not produce the expected elementwise mask.

## `diffusion.py` — missing periodic-boundary unwrapping

`get_mean_square_displacements` and `get_van_hove_self_correlation` compute
displacements directly from (possibly wrapped) positions without ever calling
`trajectory.unwrap_trajectory`. Any atom that crosses a periodic boundary
between frames will show a large spurious displacement instead of its true
small hop, corrupting MSD/diffusion-coefficient and Van Hove results for any
trajectory with real diffusive motion. A leftover unused `cell` variable in
`get_van_hove_self_correlation` suggests a PBC correction was intended but
never completed.

## `scattering.py` — `Scattering.get_structure_factor` silently zero-fills `approx_xray`

`type="approx_xray"` passes the method's input validation, but the
accumulation loop only has branches for `"neutron"` and `"xray"` — calling it
with `approx_xray` silently returns an all-zero array instead of raising or
computing anything. (`get_total_rdf`, the sibling method, does implement this
branch correctly.)

## `packing.py` — `get_random_packed` uses `print()` + `quit()` on failure

If the overlap-relaxation assertion fails, the function calls `quit()`
(`sys.exit()`), which terminates the caller's entire process/kernel rather
than raising a catchable exception. This should be replaced with a raised
exception.

## `mlip_functions.py` — deprecated ASE API

`get_pred_energy_forces` calls `a.set_calculator(calc)`, a long-deprecated ASE
API (the rest of the package, including the other function in this same
file, uses `a.calc = calc`). This may already be removed in current ASE
releases, which would make this function crash outright.

## `persistent_homology.py` — `LocalPD` and `get_local_persistence`

Quarantined: both raise `NotImplementedError` on use rather than silently
hitting the underlying `AttributeError`. They were written against
`neighborhood.get_persistence_diagram(...)` as a bound method on
GlassAtoms/ASE `Atoms`, but that functionality was moved out into the
standalone `get_persistence_diagram(atoms, ...)` function in the same module,
and the call sites were never updated. `get_local_persistence` additionally
calls `neighborhood.center()`, which does ASE's vacuum-padding centering, not
the minimum-image centering it needs. The standalone `get_persistence_diagram`
function itself is unaffected and works correctly on its own.
