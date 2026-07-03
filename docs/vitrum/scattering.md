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

- For $G(r)$ and $S(Q)$, both neutron and X-ray scattering versions are available. <br/>
    - Weighting factor for neutron diffraction: $W_{ij} = c_{i} \bar b_{i} c_{j} \bar b_{j}$ <br/>
    - Weighting factor for X-ray diffraction: $W_{ij}(Q) = c_{i}f_{i}(Q) c_{j}f_{j}(Q)$ <br/>

*Calculations are based on 'Keen, David A. "A comparison of various commonly used correlation functions for describing total scattering." *Applied Crystallography* 34, no. 2 (2001): 172-177.'* <br/>
::: vitrum.scattering
