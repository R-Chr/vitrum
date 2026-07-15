import re
from collections import deque
from fractions import Fraction
from functools import reduce
from math import gcd


def get_LAMMPS_dump_timesteps(filename: str):
    """
    Retrieves the timesteps from a LAMMPS dump file.

    Parameters:
        filename (str): The path to the LAMMPS dump file.

    Returns:
        List[int]: A list of timesteps extracted from the file.
    """
    with open(filename, encoding="utf-8") as f:
        timesteps = []
        lines = deque(f.readlines())
        if len(lines) == 0:
            return timesteps
        line = lines.popleft()
        while len(lines) > 0:
            if "ITEM: TIMESTEP" in line:
                line = lines.popleft()
                timesteps.append(int(line))
            else:
                line = lines.popleft()
    return timesteps


def correct_atom_types(atoms_list, atom_to_type_map):
    """
    Correct the atom types in a list of Atoms objects.

    Parameters:
        atoms_list (list of Atoms objects): The list of Atoms objects to correct.
        atom_to_type_map (dict): A dictionary mapping atomic numbers to atom types.

    Returns:
        atoms_list (list of Atoms objects): The corrected list of Atoms objects.
    """
    #Check if atoms_list is a list of Atoms objects
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    for atoms in atoms_list:
        corr_symbols = [atom_to_type_map[i] for i in atoms.get_atomic_numbers()]
        atoms.set_chemical_symbols(corr_symbols)

def _parse_oxide(formula):
    """'Al2O3' -> {'Al': Fraction(2), 'O': Fraction(3)} (no parentheses support)."""
    _TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
    counts = {}
    pos = 0
    for m in _TOKEN.finditer(formula):
        if m.start() != pos or not m.group(1):
            raise ValueError(f"Cannot parse formula: {formula!r}")
        el, n = m.group(1), m.group(2)
        counts[el] = counts.get(el, Fraction(0)) + (Fraction(n) if n else Fraction(1))
        pos = m.end()
    if pos != len(formula) or not counts:
        raise ValueError(f"Cannot parse formula: {formula!r}")
    return counts
  
def formula_unit(formula, integers=False, anions_last=True):
    """Convert a mol% oxide formula string into a single formula-unit string.
    Example:
    "60.2SiO2-16.0B2O3-12.6Na2O-3.8Al2O3-5.7CaO-1.7ZrO2"
    -> "Si0.602B0.32Na0.252Al0.076Ca0.057Zr0.017O2.015"   (per one formula unit)
    -> "Si602B320Na252Al76Ca57Zr17O2015"                  (smallest integers)
 
    Components may be separated by '-' or whitespace. Prefixes are treated as
    molar proportions (they need not sum to 100; the result is normalized).
 
    integers=False -> fractional coefficients per one formula unit
                      (e.g. 'Si0.602B0.32...O2.015')
    integers=True  -> smallest whole-number multiple of the formula unit
                      (e.g. 'Si602B320...O2015')
    """
    parts = [p for p in re.split(r"[-\s]+", formula.strip()) if p]
    totals, order = {}, []
    for part in parts:
        m = re.match(r"^(\d*\.?\d*)\s*(.+)$", part)
        frac = Fraction(m.group(1)) if m.group(1) else Fraction(1)
        for el, n in _parse_oxide(m.group(2)).items():
            if el not in totals:
                totals[el] = Fraction(0)
                order.append(el)
            totals[el] += frac * n
 
    total_mol = sum(Fraction(re.match(r"^(\d*\.?\d*)", p).group(1) or 1) for p in parts)
    coeffs = {el: v / total_mol for el, v in totals.items()}  # per one formula unit
 
    if anions_last:
        anions = [el for el in ("O", "S", "Se", "F", "Cl", "Br", "I") if el in order]
        order = [el for el in order if el not in anions] + anions
 
    if integers:
        lcm_den = reduce(lambda a, b: a * b // gcd(a, b),
                         (coeffs[el].denominator for el in order))
        ints = [coeffs[el].numerator * (lcm_den // coeffs[el].denominator) for el in order]
        g = reduce(gcd, ints)
        return "".join(f"{el}{n // g if n // g != 1 else ''}"
                       for el, n in zip(order, ints))
 
    def fmt(x):
        s = f"{float(x):.6g}"
        return "" if s == "1" else s
 
    return "".join(f"{el}{fmt(coeffs[el])}" for el in order)