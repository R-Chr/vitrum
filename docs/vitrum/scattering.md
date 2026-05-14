# Scattering

The scattering class contains functions for calculating scattering functions of materials, as averaged over a list of Extended ASE Atoms objects.

Included functions: <br/>
- Partial pair distribution function: $g_{ij}(r)$. `get_partial_pdf` <br/>
    - $g_{ij}(r) = \frac{n_{ij}(r)}{4 \pi r^2 dr \rho_{j}}$<br/>
    - $n_{ij}(r)$ is the number of particles of type $j$ between distance $r$ and $r + dr$ from a particle of type $i$ and $\rho_{j} = c_{j} \rho_{0}$.<br/>
    
- Total RDF $g(r)$: `get_total_rdf` <br/>
    - `total RDF? or total radius distribution function?` <br/>
    - $g(r) = \frac{\sum_{ij}W_{ij}g_{ij}(r)}{\sum_{ij}W_{ij}}$, where the $W_{ij}$ is the weighting factors in Neutron or X-ray diffraction. <br/>

- Total radius distribution function: $G(r)$. `get_reduced_pdf` <br/>
    - `Total radius distribution function? or should the name be reduced total pair distribution function?` <br/>
    - $G(r) = 4 \pi r \rho_{0} [g(r) - 1]$, where $\rho_{0}$ is the average number density. <br/>
    - **This can be used to compare with the $G(r)$ from Neutron or X-ray diffraction.** <br/>

- Total correlation function: $T(r)$. `get_T_r_pdf`<br/>
    - $T(r) = 4 \pi r\rho_{0}g(r)$, where $\rho_{0}$ is the average number density. <br/>

- Partial structure factor: $A_{ij}(Q)$. `get_partial_structure_factor`<br/>
    - $A_{ij}(Q) = 1 + \rho_{0} \int_{0}^{\infty} 4 \pi r^2 (g_{ij}(r) - 1) L(r) dr$ <br/>
    - Optional: Lorch function: $L(r) = \frac{\sin(\pi r / r_{max})}{\pi r / r_{max}}$. <br/>

- Weighted partial structure factor: $W_{ij} A_{ij}(Q)$. `get_weighted_partial_structure_factor` <br/>
    - This has the Neutron and X-ray diffraction option. <br/>

- Normalized total-scattering structure factor: $S(Q)$. `get_structure_factor` <br/>
    - $S(Q) = \frac{\sum_{ij}W_{ij}A_{ij}(Q)}{\sum_{ij}W_{ij}}$ <br/>
    - **This can be used to compare with the $S(Q)$ from experiment.** <br/>

- Calculate the running coordination number for a specific pair of elements. `get_N_running` <br/>
    - $N(r) = \int_{0}^{r} 4 \pi r^2 \rho_{j} g_{ij}(r) dr$ <br/>

- For $G(r)$ and $S(Q)$, both neutron and x-ray scattering versions are available. <br/>
    - Neutron: $W_{ij} = c_{i}b_{i} c_{j}b_{j}$ <br/>
    - X-ray: $W_{ij}(Q) = c_{i}f_{i}(Q) c_{j}f_{j}(Q)$ <br/>

*Calculations are based on 'Keen, David A. "A comparison of various commonly used correlation functions for describing total scattering." *Applied Crystallography* 34, no. 2 (2001): 172-177.'* <br/>

::: vitrum.scattering
