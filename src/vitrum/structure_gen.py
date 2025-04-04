import random
import numpy as np
from vitrum.utility import get_random_packed
import math
from pymatgen.core import Element
import tqdm


def my_round(x):
    return round(0.01 * round(x / 0.01), 3)


def balance_charge(amounts, atoms, charges_dict, max_iter=1):
    total_charge = sum(a * charges_dict[atom] for a, atom in zip(amounts, atoms))
    candidates = [(i, charges_dict[atom]) for i, atom in enumerate(atoms)]
    valid_candidates = [c for c in candidates if is_multiple(c[1], total_charge)]
    if valid_candidates:
        idx, cand_charge = min(valid_candidates, key=lambda c: abs(abs(total_charge) - abs(c[1])))
        adjustment = int(round(total_charge / cand_charge))
        amounts[idx] += -adjustment
        new_charge = sum(a * charges_dict[atom] for a, atom in zip(amounts, atoms))
    else:
        new_charge = total_charge
    return amounts, new_charge


def is_multiple(c, total, tol=1e-9):
    ratio = total / c
    return math.isclose(ratio, round(ratio), abs_tol=tol)


def choose_count(weights):
    outcomes = list(range(len(weights)))
    return random.choices(outcomes, weights=weights, k=1)[0]


def random_partition(x, total=10):
    if x <= 1:
        return [1]
    cuts = np.sort(np.random.choice(range(1, total), x - 1, replace=False))
    parts = np.diff([0] + list(cuts) + [total])
    return parts / total


def gen_random_glasses(modifiers, formers, anions, weights={}, num_structures=30, target_atoms=100, **kwargs):
    """
    Generate random glass structures from the atoms in given modifiers, formers and anions.

    Parameters:
        modifiers (list): A list of the chemical symbols of the modifiers.
        formers (list): A list of the chemical symbols of the network formers.
        anions (list): A list of the chemical symbols of the anions.
        weights (dict): A dictionary of weights for the number of modifiers, formers and anions.
        num_structures (int, optional): The number of structures to generate. Defaults to 30.
        target_atoms (int, optional): The target number of atoms in the structure. Defaults to 100.
        **kwargs: Additional keyword arguments to be passed to the get_random_packed function.

    Returns:
        compositions (list): A list of random packed structures.
    """
    charges_modifiers = {e: Element(e).oxidation_states[-1] for e in modifiers}
    charges_formers = {e: Element(e).oxidation_states[-1] for e in formers}
    charges_anions = {e: Element(e).oxidation_states[0] for e in anions}

    charges = charges_modifiers | charges_formers | charges_anions
    compositions = []
    composition_sets = set()

    num_mod_weights = weights.get("num_mod_weights", [0.5, 0.5, 0, 0])
    num_former_weights = weights.get("num_former_weights", [0.05, 0.65, 0.3, 0])
    num_anion_weights = weights.get("num_anion_weights", [0, 0.6, 0.4, 0])

    bias_modifiers = weights.get("bias_modifiers", [1 for _ in modifiers])
    bias_modifiers = bias_modifiers / np.sum(bias_modifiers)
    bias_formers = weights.get("bias_formers", [1 for _ in formers])
    bias_formers = bias_formers / np.sum(bias_formers)
    bias_anions = weights.get("bias_anions", [1 for _ in anions])
    bias_anions = bias_anions / np.sum(bias_anions)

    pbar = tqdm.tqdm(total=num_structures)
    print("Generated structures")
    while len(compositions) < num_structures:
        pbar.n = len(compositions)
        pbar.refresh()

        num_mod = choose_count(num_mod_weights)
        num_former = choose_count(num_former_weights)
        num_anion = choose_count(num_anion_weights)

        if (num_mod + num_former) == 0 or num_anion == 0:
            continue

        chosen_mods = np.random.choice(modifiers, num_mod, replace=False, p=bias_modifiers) if num_mod else []
        chosen_formers = np.random.choice(formers, num_former, replace=False, p=bias_formers) if num_former else []
        chosen_anions = np.random.choice(anions, num_anion, replace=False, p=bias_anions) if num_anion else []

        amounts = []

        mod_form_ratio = random_partition(2) if num_mod and num_former else [int(bool(num_mod)), int(bool(num_former))]
        mod_ratio = random_partition(num_mod)
        form_ratio = random_partition(num_former)
        anion_ratio = random_partition(num_anion)

        avg_mod_charge = np.sum([charges[chosen_mod] * mod_ratio[ind] for ind, chosen_mod in enumerate(chosen_mods)])
        avg_form_charge = np.sum(
            [charges[chosen_form] * form_ratio[ind] for ind, chosen_form in enumerate(chosen_formers)]
        )
        avg_anion_charge = np.sum(
            [charges[chosen_anion] * anion_ratio[ind] for ind, chosen_anion in enumerate(chosen_anions)]
        )

        avg_cation_charge = mod_form_ratio[0] * avg_mod_charge + mod_form_ratio[1] * avg_form_charge
        cation_anion_ratio = abs(avg_cation_charge) / abs(avg_anion_charge)

        if num_mod > 0:
            amounts.extend([my_round(modifier_ratio * mod_form_ratio[0]) for modifier_ratio in mod_ratio])
        if num_former > 0:
            amounts.extend([my_round(former_ratio * mod_form_ratio[1]) for former_ratio in form_ratio])
        if num_anion > 0:
            amounts.extend([my_round(ani_ratio * cation_anion_ratio) for ani_ratio in anion_ratio])
        atoms = list(chosen_mods) + list(chosen_formers) + list(chosen_anions)
        int_amounts = (np.array(amounts) * 100).astype(int)
        total_charge = sum(amt * charges[atom] for amt, atom in zip(int_amounts, atoms))

        if total_charge != 0:
            int_amounts_list = int_amounts.tolist()
            int_amounts_list, final_charge = balance_charge(int_amounts_list, atoms, charges, max_iter=1)
            if final_charge != 0:
                continue
            int_amounts = np.array(int_amounts_list)

        formula = "".join(f"{atom}{amt}" for atom, amt in zip(atoms, int_amounts))
        if formula in composition_sets:
            continue
        composition_sets.add(formula)
        rand_atoms = get_random_packed(formula, target_atoms=target_atoms, **kwargs)

        if len(rand_atoms) > 200:
            continue

        compositions.append(rand_atoms)
    return compositions
