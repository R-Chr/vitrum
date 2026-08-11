import itertools
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list
from scipy import integrate
from scipy.stats import norm
from tqdm import tqdm

from vitrum.geometry import (
    distance_matrix,
    partial_pdf,
    pdf,
    radial_bins,
    require_orthorhombic,
)

logger = logging.getLogger(__name__)


def gaussian_broadening(g_r: np.ndarray, r: np.ndarray, Q_max: float) -> np.ndarray:
    """
    Broaden the RDF to the resolution of a diffraction measurement truncated at Q_max.

    Truncating the Fourier transform at Q_max convolutes the *odd* correlation function
    r*g(r) with a Gaussian of FWHM 5.437 / Q_max, the odd kernel G(r - r') - G(r + r')
    enforcing that oddness. The broadening is therefore applied to r'*g(r') and divided
    by r to return to g(r):

        g_broad(r) = (1 / r) * Integral[ r' g(r') (G(r - r') - G(r + r')) dr' ]

    Args:
        g_r (np.ndarray): The RDF values.
        r (np.ndarray): The r values.
        Q_max (float): The maximum Q value.

    Returns:
        np.ndarray: Broadened RDF.
    """
    delta_r = r[np.newaxis, :] - r[:, np.newaxis]
    sum_r = r[np.newaxis, :] + r[:, np.newaxis]
    FWHM = 5.437 / Q_max
    sigma = FWHM / 2.355
    foubroad = r * g_r * (norm.pdf(delta_r, 0, sigma) - norm.pdf(sum_r, 0, sigma))
    dist_broad = np.trapezoid(foubroad, r) / r
    return dist_broad


class Scattering:
    """
    Class for calculating scattering functions from glass structures.
    """

    def __init__(
        self,
        atoms: list[Atoms] | Atoms,
        qmin: float = 0.5,
        qmax: float = 20.0,
        rrange: float | None = None,
        nbin: int = 500,
        neutron_scattering_coef: list[float] | None = None,
        x_ray_scattering_coef: np.ndarray | None = None,
        disable_progress: bool = False,
        use_neighborhood: bool = False,
    ):
        """
        Initializes a new instance of the class with the given atoms.

        Args:
            atoms (Union[List[Atoms], Atoms]): A list of Atoms objects or a single Atoms object.
            qmin (float, optional): The minimum q-value to use. Defaults to 0.5.
            qmax (float, optional): The maximum q-value to use. Defaults to 20.
            rrange (float, optional): The range of r-values to use. If None, defaults to min(cell_dim)/2.
            nbin (int, optional): The number of bins to use. Defaults to 500.
            neutron_scattering_coef (List[float], optional): A list of custom neutron scattering lengths. Defaults to
                None.
              If None, the default coefficients from Neutron News, Vol. 3, No. 3, 1992, pp. 29-37 are used.
            x_ray_scattering_coef (np.ndarray, optional): A list of custom x-ray scattering coefficients. Defaults to
                None.
              If None, the default coefficients from International Tables for Crystallography (2006). Vol. C. ch. 6.1,
              pp. 554-595 are used.
            disable_progress (bool, optional): Whether to disable the progress bar. Defaults to False.
        """

        if isinstance(atoms, list):
            atom_list = atoms
        else:
            atom_list = [atoms]

        self.atom_list = list(atom_list)
        script_dir = Path(__file__).parent

        # Every frame is binned with the same rrange, so every frame has to support it.
        half_min_dims = [np.min(require_orthorhombic(atom.get_cell(), "Scattering")) / 2 for atom in atom_list]
        half_min_dim = float(np.min(half_min_dims))

        if rrange:
            if rrange > half_min_dim:
                logger.warning(
                    "Specified rrange (%.2f) exceeds half the shortest cell length "
                    "(%.2f). This may violate the Minimum Image Convention.",
                    rrange,
                    half_min_dim,
                )
            self.rrange = rrange
        else:
            # Default to half the shortest cell dimension over the trajectory
            self.rrange = half_min_dim

        self.nbin = nbin
        self.xval, self.volbin = radial_bins(self.rrange, self.nbin)

        self.qval = np.linspace(qmin, qmax, self.nbin)
        self.chemical_symbols = atom_list[0].get_chemical_symbols()
        self.species = np.unique(self.chemical_symbols)
        self.species_code = {symbol: code for code, symbol in enumerate(self.species)}
        self.pairs = [pair for pair in itertools.product(self.species, repeat=2)]
        self.c = [self.chemical_symbols.count(i) / len(self.chemical_symbols) for i in self.species]

        # The composition is assumed constant across the trajectory: self.chemical_symbols,
        # self.species, self.pairs and self.c are all derived from the first frame alone.
        reference_composition = sorted(self.chemical_symbols)
        for frame_ind, atom in enumerate(atom_list[1:], start=1):
            if sorted(atom.get_chemical_symbols()) != reference_composition:
                raise ValueError(
                    f"Frame {frame_ind} has a different composition to frame 0. Scattering "
                    "assumes a fixed composition across the trajectory."
                )

        # Trajectory averages, so that NPT runs with a varying cell are weighted consistently
        # with the per-frame partial PDFs.
        volumes = np.array([atom.get_volume() for atom in atom_list])
        self.volume = float(volumes.mean())
        self.aveden = float(np.mean([len(atom) / vol for atom, vol in zip(atom_list, volumes)]))
        self.atomic_numbers = [Atom(atom).number for atom in self.species]
        self.disable_progress = disable_progress

        # Neutron
        if neutron_scattering_coef is None:
            self.scattering_lengths = pd.read_csv(script_dir / "scattering_lengths.csv", sep=";", decimal=",")
            tabulated = set(self.scattering_lengths["Isotope"])
            untabulated = [i for i in self.species if i not in tabulated]
            if untabulated:
                raise ValueError(
                    f"No tabulated neutron scattering length for {untabulated}. "
                    "Pass neutron_scattering_coef explicitly, one value per species in "
                    f"{list(self.species)}."
                )
            self.b = np.array(
                [self.scattering_lengths[self.scattering_lengths["Isotope"] == i]["b"] for i in self.species]
            ).flatten()
        else:
            self.b = np.asarray(neutron_scattering_coef, dtype=float)

        self.cb = [i * j for i, j in zip(self.c, self.b)]
        self.timesby = [pair[0] * pair[1] for pair in itertools.product(self.cb, repeat=2)]

        # X-ray
        if x_ray_scattering_coef is None:
            x_ray_scattering_coef_df = pd.read_csv(script_dir / "x_ray_scattering_factor_coefficients.csv", sep=",")
            x_ray_scattering_coef_arr = np.array(
                [x_ray_scattering_coef_df[x_ray_scattering_coef_df["Element"] == i] for i in self.species]
            ).reshape([len(self.species), 10])
        else:
            x_ray_scattering_coef_arr = x_ray_scattering_coef

        # Column 0 is the element symbol, so the array read from the CSV has dtype object.
        self.x_ray_a = x_ray_scattering_coef_arr[:, [1, 3, 5, 7]].astype(float)
        self.x_ray_b = x_ray_scattering_coef_arr[:, [2, 4, 6, 8]].astype(float)
        self.x_ray_c = x_ray_scattering_coef_arr[:, [9]].astype(float)

        self.f_i = list(self._form_factors(self.qval))

        self.xray_cb = [i * j for i, j in zip(self.c, self.f_i)]
        self.xray_timesby = [pair[0] * pair[1] for pair in itertools.product(self.xray_cb, repeat=2)]

        self.approx_xray_cb = np.array([i * j for i, j in zip(self.c, self.atomic_numbers)])
        self.approx_xray_timesby = np.array(
            [pair[0] * pair[1] for pair in itertools.product(self.approx_xray_cb, repeat=2)]
        )

        self._weights = {
            "neutron": np.asarray(self.timesby, dtype=float),
            "xray": np.asarray(self.xray_timesby, dtype=float),
            "approx_xray": np.asarray(self.approx_xray_timesby, dtype=float),
        }

        if use_neighborhood:
            self.partial_pdfs = self.calculate_partial_pdfs_neighborhood()
        else:
            self.partial_pdfs = self.calculate_partial_pdfs()

    def _form_factors(self, q: np.ndarray) -> np.ndarray:
        """Cromer-Mann x-ray form factors f_i(Q) for every species, shape (n_species, nq)."""
        s2 = (np.asarray(q, dtype=float) / (4 * math.pi)) ** 2
        return (self.x_ray_a[:, :, None] * np.exp(-self.x_ray_b[:, :, None] * s2)).sum(axis=1) + self.x_ray_c

    def _normalized_weights(self, type: str) -> np.ndarray:
        """
        Weights W_ij / sum_ij W_ij for every ordered pair in self.pairs.

        Args:
            type (str): The weighting scheme, one of "neutron", "xray" or "approx_xray".

        Returns:
            np.ndarray: Shape (n_pairs,) for the Q-independent schemes, and (n_pairs, nbin)
                for "xray", whose f_i(Q) are tabulated on self.qval.

        Raises:
            ValueError: If type is not a known weighting scheme.
        """
        if type not in self._weights:
            raise ValueError(f"Invalid type {type!r}. Choose one of {sorted(self._weights)}.")
        w = self._weights[type]
        return w / w.sum(axis=0)

    def calculate_partial_pdfs(self) -> np.ndarray:
        """
        Calculate partial PDFs for all pairs from the full distance matrix.

        Builds an N x N distance matrix per frame, so time and memory both scale as
        O(N^2) regardless of `rrange`. This is the faster backend at the default
        `rrange` of half the shortest cell length, where a neighbour list holds nearly
        every pair anyway; see `calculate_partial_pdfs_neighborhood` for the small-cutoff
        alternative.

        Returns:
            np.ndarray: Array of partial PDFs.
        """
        pdf_sum = np.zeros((len(self.pairs), self.nbin))
        n_frames = len(self.atom_list)

        for atom in tqdm(self.atom_list, disable=self.disable_progress):
            distances = distance_matrix(atom, "Scattering")
            symbols = np.array(atom.get_chemical_symbols())
            volume = atom.get_volume()

            for pair_ind, pair in enumerate(self.pairs):
                _, current_pdf = partial_pdf(distances, symbols, volume, pair, self.rrange, self.nbin)
                pdf_sum[pair_ind, :] += current_pdf
        return pdf_sum / n_frames

    def calculate_partial_pdfs_neighborhood(self) -> np.ndarray:
        """
        Calculate partial PDFs from a neighbour list, avoiding the full distance matrix.

        Cost scales with the number of pairs within `rrange` rather than with N^2, so this
        wins only when `rrange` is small compared with the cell: at the default `rrange` of
        half the shortest cell length the neighbour sphere covers much of the cell and this
        is slower than `calculate_partial_pdfs`. It is the memory-frugal option either way,
        since no N x N matrix is built.

        Returns:
            np.ndarray: Array of partial PDFs.
        """
        all_frame_data: dict[tuple[str, str], list[np.ndarray]] = {pair: [] for pair in self.pairs}
        n_species = len(self.species)
        for atom in tqdm(self.atom_list, disable=self.disable_progress):
            symbols = np.array(atom.get_chemical_symbols())
            volume = atom.get_volume()
            i_list, j_list, d_list = neighbor_list("ijd", a=atom, cutoff=self.rrange)
            # Label each neighbour pair by an unordered pair of species codes, so the frame
            # splits into per-pair distance arrays with one boolean mask each.
            codes = np.array([self.species_code[s] for s in symbols])
            code_i, code_j = codes[i_list], codes[j_list]
            pair_keys = np.minimum(code_i, code_j) * n_species + np.maximum(code_i, code_j)

            for pair in self.pairs:
                el1, el2 = pair
                # g_ij == g_ji, so both orderings of a cross pair share a key.
                low, high = sorted((self.species_code[el1], self.species_code[el2]))
                distances = d_list[pair_keys == low * n_species + high]
                n1 = int(np.sum(symbols == el1))
                n2 = int(np.sum(symbols == el2))
                # `neighbor_list` reports both (i, j) and (j, i), so a cross pair's distances
                # appear twice. A like pair's n1 * (n1 - 1) ordered pairs already account for it.
                if el1 == el2:
                    n_pairs = n1 * (n1 - 1)
                else:
                    n_pairs = 2 * n1 * n2
                _, current_pdf = pdf(
                    distances,
                    volume,
                    self.rrange,
                    self.nbin,
                    n_pairs=n_pairs,
                    exclude_self=False,  # a neighbour list never contains self-distances
                )
                all_frame_data[pair].append(current_pdf)
        pdfs = np.zeros((len(self.pairs), self.nbin))

        for pair_ind, pair in enumerate(self.pairs):
            if all_frame_data[pair]:
                pdfs[pair_ind] = np.mean(all_frame_data[pair], axis=0)
            else:
                pdfs[pair_ind] = np.zeros(self.nbin)

        return pdfs

    def get_partial_pdf(self, pair: tuple[str, str]) -> np.ndarray:
        """
        Get the partial probability density function (PDF) of a given pair of target atoms.

        Args:
            pair (Tuple[str, str]): A tuple of two elements representing the target atoms. Example: ('Si', 'O')

        Returns:
            np.ndarray: An array of shape (nbin,) containing the PDF values.
        """
        return self.partial_pdfs[self.pairs.index(pair)]

    def _xray_total_rdf(self, lorch: bool = False) -> np.ndarray:
        """
        Calculate the x-ray weighted total radial distribution function G^X(r).

        Keen (2001) eqs 57-61, with eq 61 truncated at the instance's qmax:

            f_ij(Q)   = f_i(Q) f_j(Q) / [sum_k c_k f_k(Q)]^2                         (57)
            j_ij(r)   = (1 / pi) Integral[ f_ij(Q) M(Q) cos(Qr) dQ ]                 (61)
            g^X_ij(r) = (1 / r) Integral[ r' (g_ij(r') - 1) j_ij(r - r') dr' ]       (60)
            G^X(r)    = sum_ij c_i c_j g^X_ij(r)                                     (59)

        lorch sets M(Q) to the Lorch function, tapering the integrand to zero at qmax
        instead of cutting it off. Returns G^X(r); callers wanting G'(r) add 1.
        """
        q_max = float(self.qval[-1])
        nq = int(np.ceil(30 * q_max * self.rrange / math.pi)) + 2
        q = np.linspace(0.0, q_max, nq)
        f = self._form_factors(q)
        f_norm = np.square(np.asarray(self.c) @ f)  # [sum_k c_k f_k(Q)]^2.
        mod = np.sinc(q / q_max) if lorch else 1.0

        dr = self.rrange / self.nbin
        s = np.arange(2 * self.nbin) * dr
        cos_qs = np.cos(np.outer(s, q))
        delta_r = np.abs(self.xval[np.newaxis, :] - self.xval[:, np.newaxis])
        sum_r = self.xval[np.newaxis, :] + self.xval[:, np.newaxis]

        # j_ij == j_ji, so each unordered pair is transformed once and reused for both orderings.
        kernels: dict[tuple[int, int], np.ndarray] = {}
        g_x = np.zeros(self.nbin)
        for pair in self.pairs:
            i, j = self.species_code[pair[0]], self.species_code[pair[1]]
            key = (min(i, j), max(i, j))
            if key not in kernels:
                kernels[key] = np.trapezoid(f[i] * f[j] / f_norm * mod * cos_qs, q) / math.pi  # eqs 57, 61
            j_ij = kernels[key]
            # eq 60; r'(g-1) extended odd, folding r' < 0 onto the j_ij(r + r') term.
            d = self.xval * (self.get_partial_pdf(pair) - 1.0)
            conv = np.trapezoid(d * (np.interp(delta_r, s, j_ij) - np.interp(sum_r, s, j_ij)), self.xval)
            g_x = g_x + self.c[i] * self.c[j] * conv / self.xval  # eq 59
        return g_x

    def get_total_rdf(self, type: str = "neutron", broaden: bool | float = False, lorch: bool = False) -> np.ndarray:
        """
        Calculate the total RDF for a given number of bins and range.

        Args:
            type (str, optional): The type of structure factor to calculate, one of "neutron",
                "xray" or "approx_xray". Defaults to "neutron".
            broaden (Union[bool, int, float], optional): If a number, apply Gaussian broadening
                to the RDF at that maximum Q value. Defaults to False. type="xray" is already
                broadened to qmax by the transform it is built from.
            lorch (bool, optional): If True, apply the Lorch modification function to the
                transform behind type="xray", suppressing truncation ripples at the cost of
                real-space resolution. Defaults to False.

        Returns:
            np.ndarray: An array of shape (nbin,) containing the total RDF values.

        Raises:
            ValueError: If type is invalid, broaden is invalid, or lorch is used with a
                weighting that involves no transform.

        Note:
            type="xray" is built from a transform truncated at qmax and ripples below the
            first bond; raise qmax or pass lorch=True. See the scattering docs.
        """
        weights = self._normalized_weights(type)  # validates `type` before any work is done
        if type == "xray":
            gr_tot = 1.0 + self._xray_total_rdf(lorch=lorch)
        else:
            if lorch:
                raise ValueError(
                    f"lorch=True is meaningless for type={type!r}, which weights the partial "
                    "PDFs directly and involves no Fourier transform to modify. It applies "
                    "only to type='xray'."
                )
            gr_tot = weights @ self.partial_pdfs
        if broaden:
            if isinstance(broaden, (int, float)) and not isinstance(broaden, bool):
                # bool check needed because bool is subclass of int in Python
                Q_max = float(broaden)
            else:
                raise ValueError("broaden must be a number (Q_max) to apply broadening.")

            gr_tot = gaussian_broadening(gr_tot, self.xval, Q_max)

        return gr_tot

    def get_partial_structure_factor(self, target_atoms: tuple[str, str], lorch: bool = False) -> np.ndarray:
        """
        Calculate the partial structure factor for a given target atoms within a specified range.

        Args:
            target_atoms (Tuple[str, str]): A tuple of two elements representing the target atoms.
            lorch (bool, optional): If True, apply Lorch correction to the structure factor.

        Returns:
            np.ndarray: An array of shape (nbin,) containing the partial structure factor.
        """

        pdf_val = self.get_partial_pdf(pair=target_atoms)
        # np.sinc(x) is sin(pi x)/(pi x), which is 1 at x = 0, so r = 0 needs no special case.
        kernel = np.sinc(np.outer(self.qval, self.xval) / np.pi)
        integrand = 4 * math.pi * self.xval**2 * (pdf_val - 1) * kernel
        if lorch:
            integrand = integrand * np.sinc(self.xval / self.rrange)
        return 1 + self.aveden * np.trapezoid(integrand, self.xval)

    def get_weighted_partial_structure_factors(
        self,
        type: str = "neutron",
        lorch: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Calculate weighted partial structure factors W_ij * S_ij(Q) for all
        unique element pairs.

        Weights follow the same definition as get_structure_factor():
            neutron: W_ij = c_i*b_i * c_j*b_j / (sum_k c_k*b_k)^2
            xray:    W_ij(Q) = c_i*f_i(Q) * c_j*f_j(Q) / (sum_k c_k*f_k(Q))^2

        Cross terms (i != j) are merged: S_ij = S_ji, so their weights are
        multiplied by 2 and only one label (e.g. "Si-O") is returned.

        Summing the returned values gives get_structure_factor(type=type).

        Args:
            type (str): Weighting scheme, one of "neutron", "xray" or "approx_xray".
                Defaults to "neutron".
            lorch (bool): If True, apply Lorch modification function to reduce
                truncation ripples. Passed through to get_partial_structure_factor().
                Defaults to False.

        Returns:
            Dict[str, np.ndarray]: Dict mapping pair label (e.g. "Si-O") to
                W_ij * S_ij(Q), each of shape (nbin,).

        Raises:
            ValueError: If type is not a known weighting scheme.
        """
        weights = self._normalized_weights(type)

        unique_pairs = list(itertools.combinations_with_replacement(self.species, 2))

        weighted_partials: dict[str, np.ndarray] = {}

        for pair in unique_pairs:
            label = f"{pair[0]}-{pair[1]}"

            try:
                idx = self.pairs.index(pair)
            except ValueError:
                idx = self.pairs.index((pair[1], pair[0]))

            partial_sq = self.get_partial_structure_factor(target_atoms=(pair[0], pair[1]), lorch=lorch)

            multiplier = 1.0 if pair[0] == pair[1] else 2.0
            w_sij = np.asarray(multiplier * weights[idx] * partial_sq, dtype=float)
            weighted_partials[label] = w_sij

        return weighted_partials

    def get_structure_factor(self, type: str = "neutron", lorch: bool = False) -> np.ndarray:
        """
        Calculate the total structure factor.

        Args:
            type (str, optional): The type of structure factor to calculate, one of "neutron",
                "xray" or "approx_xray". Defaults to "neutron".
            lorch (bool, optional): whether to apply lorch correction.

        Returns:
            np.ndarray: An array of shape (nbin,) containing the total structure factor.

        Raises:
            ValueError: If type is not a known weighting scheme.
        """
        weights = self._normalized_weights(type)

        # S_ij == S_ji, so each unordered pair is transformed once and reused for both
        # orderings.
        transforms = {}
        S_q_tot = np.zeros(self.nbin)
        for ind, pair in enumerate(self.pairs):
            key = tuple(sorted((self.species_code[pair[0]], self.species_code[pair[1]])))
            if key not in transforms:
                transforms[key] = self.get_partial_structure_factor(target_atoms=(pair[0], pair[1]), lorch=lorch)
            S_q_tot = S_q_tot + weights[ind] * transforms[key]
        return S_q_tot

    def get_T_r_pdf(self, type: str = "neutron", broaden: bool | float = False, lorch: bool = False) -> np.ndarray:
        """
        Calculate the total correlation function T(r).

        T(r) = 4 * pi * r * rho_0 * g(r)
        where rho_0 is the average number density.

        Args:
            type (str, optional): The type of scattering, one of "neutron", "xray" or
                "approx_xray". Defaults to "neutron".
            broaden (Union[bool, int, float], optional): Broadening parameter. Defaults to False.
            lorch (bool, optional): Apply the Lorch modification function; type="xray" only.
                Passed through to get_total_rdf(). Defaults to False.

        Returns:
            np.ndarray: The T(r) function values.
        """
        rdf = self.get_total_rdf(type=type, broaden=broaden, lorch=lorch)
        return 4 * math.pi * self.xval * self.aveden * rdf

    def get_reduced_pdf(self, type: str = "neutron", broaden: bool | float = False, lorch: bool = False) -> np.ndarray:
        """
        Get reduced PDF G(r), also written D(r).

        D(r) = 4 * pi * r * rho_0 * [G'(r) - 1]

        Its factor of r cancels the 1/r of Keen eq 60, so for type="xray" this is the better
        function to read at small r.

        Args:
            type (str, optional): The type of scattering, one of "neutron", "xray" or
                "approx_xray". Defaults to "neutron".
            broaden (Union[bool, int, float], optional): Broadening parameter. Defaults to False.
            lorch (bool, optional): Apply the Lorch modification function; type="xray" only.
                Passed through to get_total_rdf(). Defaults to False.

        Returns:
            np.ndarray: The D(r) function values.
        """
        t_r = self.get_T_r_pdf(type=type, broaden=broaden, lorch=lorch)
        return t_r - 4 * math.pi * self.xval * self.aveden

    def get_N_running(self, pair: tuple[str, str]) -> np.ndarray:
        """
        Calculate the running coordination number for a specific pair of elements.

        This is the integral of the partial RDF up to distance r:
        N(r) = Integral(4 * pi * rho_j * g_ij(r) * r^2 dr)

        Args:
            pair (Tuple[str, str]): Tuple of atomic symbols (e.g., ("Si", "O")).

        Returns:
            np.ndarray: The running coordination number as a function of r.
        """
        pair_pdf = self.get_partial_pdf(pair)
        c_j = self.c[list(self.species).index(pair[1])]
        n_v = c_j * self.aveden
        integrand = 4 * np.pi * n_v * pair_pdf * self.xval**2
        return integrate.cumulative_trapezoid(integrand, self.xval, initial=0.0)
