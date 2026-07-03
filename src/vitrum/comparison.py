import numpy as np
from scipy.interpolate import interp1d


def r_chi(function_1, function_2, x_min=0, x_max=np.inf, steps=100):
    """
    Calculate the Wright coefficient (https://doi.org/10.1016/0022-3093(93)90232-M) between two functions

    Parameters:
        function_1 (dict): Dictionary with keys 'x' and 'y' representing the first function, usually from simulations
        function_2 (dict): Dictionary with keys 'x' and 'y' representing the second function, usually from experimental meassurements.

    Returns:
        rchi (float): Wright coefficient, a measure of similarity between the two functions.
    """
    # Determine the overlapping x-range
    min_x_val = np.max([np.min(function_1["x"]), np.min(function_2["x"]), x_min])
    max_x_val = np.min([np.max(function_1["x"]), np.max(function_2["x"]), x_max])

    if min_x_val >= max_x_val:
        raise ValueError("No overlapping x-range between the two functions.")

    # Create common x-axis over the overlap region
    common_x = np.linspace(min_x_val, max_x_val, steps)

    # Interpolate both functions onto the common x-axis
    interp_f1 = interp1d(function_1["x"], function_1["y"], kind="linear", bounds_error=False, fill_value=0)
    interp_f2 = interp1d(function_2["x"], function_2["y"], kind="linear", bounds_error=False, fill_value=0)

    y1 = interp_f1(common_x)
    y2 = interp_f2(common_x)

    # Calculate the r-chi (Wright coefficient)
    numerator = np.sum((y1 - y2) ** 2)

    denominator = np.sum(y2**2)
    rchi = np.sqrt(numerator / denominator)

    return rchi, common_x, y1, y2
