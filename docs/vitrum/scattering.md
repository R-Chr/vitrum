# Scattering

The scattering class contains functions for calculating scattering functions of materials, as averaged over a list of Extended ASE Atoms objects.

Included functions: <br/>

- Partial pair distribution function: $g_{ij}(r)$: `get_partial_pdf` <br/>
    - $g_{ij}(r) = \frac{n_{ij}(r)}{4 \pi r^2 dr \rho_{j}}$<br/>
    - $n_{ij}(r)$ is the number of particles of type $j$ between distance $r$ and $r + dr$ from a particle of type $i$ and $\rho_{j} = c_{j} \rho_{0}$.<br/>
    
- Normalized total radial distribution function $G'(r)$: `get_total_rdf` <br/>

    - $G'(r) = \frac{\sum_{i,j=1}^{n} W_{ij} g_{ij}(r)}{(\sum_{i=1}^{n} c_{i} \bar b_{i})^2}$. <br/>

-  Differential correlation function: $D(r)$: `get_reduced_pdf` <br/>
    - $D(r) = 4 \pi r \rho_{0} [G'(r) - 1]$, where $\rho_{0}$ is the average number density. <br/>
    - This function is occasionally referred to as $G(r)$ (reduced pair distribution function) in literature. <br/>

- Total correlation function: $T(r)$: `get_T_r_pdf`<br/>
    - $T(r) = 4 \pi r\rho_{0} G'(r)$, where $\rho_{0}$ is the average number density. <br/>

- Partial structure factor: $A_{ij}(Q)$: `get_partial_structure_factor`<br/>
    - $A_{ij}(Q) = 1 + \rho_{0} \int_{0}^{\infty} 4 \pi r^2 (g_{ij}(r) - 1) L(r) dr$ <br/>
    - Optional: Lorch function: $L(r) = \frac{\sin(\pi r / r_{max})}{\pi r / r_{max}}$. <br/>

- Weighted partial structure factor: $W_{ij} A_{ij}(Q)$: `get_weighted_partial_structure_factors` <br/>

- Normalized total-scattering structure factor: $S(Q)$: `get_structure_factor` <br/>
    - $S(Q) = \frac{\sum_{i,j=1}^{n} W_{ij}A_{ij}(Q)}{(\sum_{i=1}^{n} c_{i} \bar b_{i})^2}$ <br/>

- Calculate the running coordination number for a specific pair of elements: `get_N_running` <br/>
    - $N(r) = \int_{0}^{r} 4 \pi r^2 \rho_{j} g_{ij}(r) dr$ <br/>

- For $G(r)$ and $S(Q)$, both neutron and X-ray scattering versions are available, selected with `type=`. <br/>
    - Weighting factor for neutron diffraction: $W_{ij} = c_{i} \bar b_{i} c_{j} \bar b_{j}$ (`type="neutron"`) <br/>
    - Weighting factor for X-ray diffraction: $W_{ij}(Q) = c_{i}f_{i}(Q) c_{j}f_{j}(Q)$ (`type="xray"`) <br/>
    - `type="approx_xray"` replaces $f_{i}(Q)$ with the atomic number $Z_{i} = f_{i}(0)$, giving a Q-independent weight. <br/>

In $Q$-space the X-ray weights can be applied directly, but $f_{i}(Q)$ cannot be taken outside the sum in $r$-space. The Q-dependence instead enters as a real-space convolution kernel: <br/>

- $f_{ij}(Q) = f_{i}(Q)f_{j}(Q) / [\sum_{i=1}^{n} c_{i}f_{i}(Q)]^2$ <br/>
- $j_{ij}(r) = \frac{1}{\pi} \int_{0}^{\infty} f_{ij}(Q) \cos(Qr) dQ$ <br/>
- $g^{X}_{ij}(r) = \frac{1}{r} \int_{-\infty}^{\infty} r' [g_{ij}(r') - 1] \, j_{ij}(r - r') dr'$, where $r'[g_{ij}(r') - 1]$ is extended as an odd function of $r'$. <br/>
- $G^{X}(r) = \sum_{i,j=1}^{n} c_{i}c_{j}g^{X}_{ij}(r)$, and `get_total_rdf(type="xray")` returns $G'(r) = 1 + G^{X}(r)$ to match the other weightings. <br/>

$f_{ij}(Q)$ tends to a constant at high $Q$, so the $j_{ij}(r)$ integral does not converge and is truncated at the instance's `qmax`, exactly as a measurement is. `get_total_rdf(type="xray")` is therefore already broadened to that resolution, and passing `broaden` on top of it is usually not what you want. The neutron and `approx_xray` RDFs involve no transform and are not truncated.

## Choosing `qmax` for the X-ray RDF

Cutting the $j_{ij}(r)$ integral off sharply at `qmax` leaves ripples in $G'(r)$, and at low `qmax` they are large enough to be mistaken for structure. Three things help, in the order worth trying them:

- **Raise `qmax`.** Free in a simulation, and the only option that suppresses ripples *and* sharpens genuine features. Match a measurement's `qmax` only when comparing to that measurement.
- **`lorch=True`**, on `get_total_rdf`, `get_T_r_pdf` and `get_reduced_pdf`. $M(Q) = \frac{\sin(\pi Q / Q_{max})}{\pi Q / Q_{max}}$ multiplies the $j_{ij}(r)$ integrand, tapering it to zero at `qmax` rather than cutting it off.
- **Read $D(r)$ rather than $G'(r)$ at small $r$.** `get_reduced_pdf` carries a factor of $r$ that cancels the $1/r$ amplifying the ripple as $r \to 0$.

## Choosing `rrange`

`rrange` sets how far $g(r)$ is tabulated, and it is the single number that decides whether a large structure is analysable at all: the number of pairs inside the range grows as $r_{max}^3$.

When `rrange` is not given it defaults to **half the shortest perpendicular cell width, capped at 20 Å**. An explicitly passed `rrange` is not subject to the 20 Å cap, but it *is* subject to the hard limit below.

### `rrange` cannot exceed half the shortest perpendicular width

Past that radius, every pair still to be counted is a periodic replica of a pair already counted, and there is no honest $g(r)$ to report — so `Scattering` raises `ValueError` rather than returning one. To tabulate further, run a larger cell.


## Peak metrics

The derived scalars usually quoted from these functions — the bond length and its static disorder from the first peak of a partial $g(r)$, and the position, width and intensity of the first sharp diffraction peak of $S(Q)$ — are the same measurement made on two different arrays, so one function in `vitrum.geometry` does both:

```python
from vitrum.geometry import peak_metrics

sc = Scattering(atoms)

# bond length and first-peak width, from the partial PDF
r_1, width, height = peak_metrics(sc.xval, sc.get_partial_pdf(("Si", "O")))

# FSDP: 2*pi/Q_1 is the quasi-periodicity it corresponds to
q_1, q_width, intensity = peak_metrics(sc.qval, sc.get_structure_factor(), window=(1.0, 3.0))
```

The `window` is not optional in practice for $S(Q)$: below the FSDP a simulated structure factor is small and noisy, and its noise has local maxima that come first. Widths are returned as NaN where the function does not fall to half maximum on both sides of the peak, which is the sign that the window is too tight.

*Calculations are based on 'Keen, David A. "A comparison of various commonly used correlation functions for describing total scattering." *Applied Crystallography* 34, no. 2 (2001): 172-177.'* <br/>
::: vitrum.scattering
