import random
import numpy as np
from vitrum.utility import get_random_packed
import math


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


def gen_random_glasses(modifiers, formers, anions, num_structures=30, target_atoms=100, mp_api_key=None, **kwargs):
    """
    Generate random glass structures from the atoms in given modifiers, formers and anions.

    Parameters:
        modifiers (dict): A dictionary mapping the chemical symbols of the modifiers with their corresonding charge.
        formers (dict): A dictionary mapping the chemical symbols of the network formers with their corresonding charge.
        anions (dict): A dictionary mapping the chemical symbols of the anions with their corresonding charge.
        target_atoms (int, optional): The target number of atoms in the structure. Defaults to 100.
        mp_api_key (str, optional): The API key for the Materials Project. Required if density is not provided.
        **kwargs: Additional keyword arguments to be passed to the get_random_packed function.

    Returns:
        compositions (list): A list of random packed structures.
    """

    charges = modifiers | formers | anions
    RATIOS = np.linspace(0.2, 0.8, 4)
    compositions = []
    composition_sets = set()

    def choose_count(weights):
        outcomes = list(range(len(weights)))
        return random.choices(outcomes, weights=weights, k=1)[0]

    while len(compositions) < num_structures:
        mod_weights = [0.5, 0.5, 0, 0]
        former_weights = [0.1, 0.6, 0.3, 0]
        anion_weights = [0, 0.6, 0.4, 0]

        num_mod = choose_count(mod_weights)
        num_former = choose_count(former_weights)
        num_anion = choose_count(anion_weights)

        chosen_mods = random.sample(list(modifiers.keys()), num_mod) if num_mod else []
        chosen_formers = random.sample(list(formers.keys()), num_former) if num_former else []
        chosen_anions = random.sample(list(anions.keys()), num_anion) if num_anion else []

        if (num_mod + num_former) == 0 or num_anion == 0:
            continue

        if num_mod and num_former:
            mod_form_ratio = random.choice(RATIOS)
        elif num_mod:
            mod_form_ratio = 1
        else:
            mod_form_ratio = 0

        amounts = []

        if num_mod == 1:
            mod_amount = my_round(mod_form_ratio)
            amounts.append(mod_amount)
            avg_mod_charge = modifiers[chosen_mods[0]]
        else:
            avg_mod_charge = 0

        if num_former == 1:
            form_amount = my_round(1 - mod_form_ratio)
            amounts.append(form_amount)
            avg_form_charge = formers[chosen_formers[0]]
        elif num_former == 2:
            form_ratio = random.choice(RATIOS)
            amt1 = my_round(form_ratio * (1 - mod_form_ratio))
            amt2 = my_round((1 - form_ratio) * (1 - mod_form_ratio))
            amounts.extend([amt1, amt2])
            avg_form_charge = formers[chosen_formers[0]] * form_ratio + formers[chosen_formers[1]] * (1 - form_ratio)
        else:
            avg_form_charge = 0

        if num_mod and num_former:
            avg_cation_charge = mod_form_ratio * avg_mod_charge + (1 - mod_form_ratio) * avg_form_charge
        elif num_mod:
            avg_cation_charge = avg_mod_charge
        else:
            avg_cation_charge = avg_form_charge

        if num_anion == 1:
            anion = chosen_anions[0]
            avg_anion_charge = anions[anion]
            cation_anion_ratio = abs(avg_cation_charge) / abs(avg_anion_charge)
            anion_amount = my_round(cation_anion_ratio)
            amounts.append(anion_amount)
        elif num_anion == 2:
            anion_ratio = random.choice(RATIOS)
            amt1, amt2 = None, None
            an1, an2 = chosen_anions[0], chosen_anions[1]
            avg_anion_charge = anions[an1] * anion_ratio + anions[an2] * (1 - anion_ratio)
            cation_anion_ratio = abs(avg_cation_charge) / abs(avg_anion_charge)
            amt1 = my_round(anion_ratio * cation_anion_ratio)
            amt2 = my_round((1 - anion_ratio) * cation_anion_ratio)
            amounts.extend([amt1, amt2])
        else:
            continue

        atoms = chosen_mods + chosen_formers + chosen_anions
        int_amounts = (np.array(amounts) * 100).astype(int)

        formula = "".join(f"{atom}{amt}" for atom, amt in zip(atoms, int_amounts))
        if formula in composition_sets:
            continue
        composition_sets.add(formula)

        total_charge = sum(amt * charges[atom] for amt, atom in zip(int_amounts, atoms))

        if total_charge != 0:
            int_amounts_list = int_amounts.tolist()
            int_amounts_list, final_charge = balance_charge(int_amounts_list, atoms, charges, max_iter=1)
            if final_charge != 0:
                continue
            int_amounts = np.array(int_amounts_list)

        rand_atoms = get_random_packed(formula, target_atoms=target_atoms, mp_api_key=mp_api_key, **kwargs)

        if len(rand_atoms) > 200:
            continue

        compositions.append(rand_atoms)
