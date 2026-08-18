# Structure generation

The [`GlassGenerator`][vitrum.structure_gen.GlassGenerator] class unifies several
strategies for sampling glass compositions and, optionally, realizing them as
random-packed atomic structures for glass simulations and materials discovery.

The workflow is always the same three steps: build a generator, call `sample()`
to get a [`Compositions`][vitrum.structure_gen.Compositions] object (a thin
wrapper around a `pandas.DataFrame`, accessible as `.df`), then optionally call
its `get_structures()` method to pack the ones you want.

## Chemical system modes

A generator works in one of two modes, chosen by how you describe the system.
There are no built-in chemistry defaults — you must specify the system explicitly:

- **Unit mode** — pass `units` (a list of oxide/component formulas). Samples are
  mole fractions per unit. Schemes: `"sobol"`, `"lhs"`, `"random"` (stochastic
  simplex sampling) and `"grid"` (regular composition grid).
- **Elemental mode** — pass `elements`, a dict grouping element symbols by role
  (`"formers"` / `"modifiers"` / `"anions"`). Samples are charge-neutral atomic
  fractions. Scheme: `"random"`.

Exactly one of `units` or `elements` must be supplied. `"random"` is valid in
both modes; the generator's mode selects which behaviour applies.

## Examples

Stochastic simplex sampling over oxides (`"sobol"`, `"lhs"` for Latin hypercube,
or `"random"`). Pass `network_formers` to enable the `min_former_sum` /
`require_former` constraints:

```python
from vitrum.structure_gen import GlassGenerator

gen = GlassGenerator(
    units=["SiO2", "B2O3", "Na2O", "CaO"],
    network_formers={"SiO2", "B2O3"},
    min_former_sum=0.4,
    seed=0,
)
comps = gen.sample("lhs", n=200)   # Compositions wrapping a DataFrame (comps.df)
structures = comps.get_structures(target_atoms=100, density=2.5)
```

Regular composition grid (`"grid"`, exhaustive — `n` is ignored):

```python
gen = GlassGenerator(units=["SiO2", "Na2O"])
comps = gen.sample("grid", spacing=10)   # 0/10/20/.../100 % of each unit
```

Random charge-balanced glasses (`"random"`, elemental mode):

```python
gen = GlassGenerator(
    elements={"formers": ["Si", "B"], "modifiers": ["Na", "Ca"], "anions": ["O"]},
    seed=0,
)
comps = gen.sample("random", n=30)
formulas = comps.to_formulas()           # e.g. ['Si12B4Na6O...', ...]
pmg = comps.to_pymatgen()                # list of pymatgen Composition objects
```

The module-level [`gen_random_glasses`][vitrum.structure_gen.gen_random_glasses]
function is retained as a backwards-compatible wrapper around the `"random"`
scheme.

::: vitrum.structure_gen
