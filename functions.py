import pandas as pd
from ase import io
import numpy as np
from ase import Atoms
import dionysus
import diode


class glass_Atoms(Atoms):
    #def __init__(self):
    #    super().__init__()

    def get_dist(self):
        dim = np.diagonal(self.get_cell())
        positions = self.get_positions()
        x_dif = np.abs(positions[:, 0][np.newaxis, :] - positions[:, 0][:, np.newaxis])
        y_dif = np.abs(positions[:, 1][np.newaxis, :] - positions[:, 1][:, np.newaxis])
        z_dif = np.abs(positions[:, 2][np.newaxis, :] - positions[:, 2][:, np.newaxis])
        x_dif = np.where(x_dif > 0.5 * dim[0], np.abs(x_dif - dim[0]), x_dif)
        y_dif = np.where(y_dif > 0.5 * dim[1], np.abs(y_dif - dim[1]), y_dif)
        z_dif = np.where(z_dif > 0.5 * dim[2], np.abs(z_dif - dim[2]), z_dif)
        i_i = np.sqrt(x_dif ** 2 + y_dif ** 2 + z_dif ** 2)
        return i_i

    def get_pdf(self, target_atoms, rrange=10, nbin=100):
        types = self.get_atomic_numbers()
        distances = self.get_dist()
        atom_1 = np.where(types == target_atoms[0])[0]
        atom_2 = np.where(types == target_atoms[1])[0]
        dist_list = distances[np.ix_(atom_1, atom_2)]
        edges = np.linspace(0, rrange, nbin+1)
        xval = edges[1:]-0.5*(rrange/nbin)
        volbin = []
        for i in range(nbin):
            vol = ((4/3)*np.pi*(edges[i+1])**3)-((4/3)*np.pi*(edges[i])**3)
            volbin.append(vol)

        h, bin_edges = np.histogram(dist_list, bins=nbin, range=(0, rrange))
        h[0] = 0
        pdf = (h/volbin)/(dist_list.shape[0]*dist_list.shape[1]/self.get_volume())
        return xval, pdf

    def get_persistence_diagram(self, dimension=1, weights=None):
        coord = self.get_positions()
        cell = self.get_cell()
        data = np.column_stack([self.get_chemical_symbols(),coord])
        dfpoints = pd.DataFrame(data, columns=["Atom", "x", "y", "z"])
        chem_species = np.unique(self.get_chemical_symbols())

        if weights is None:
            radii = [0 for i in chem_species]
        elif isinstance(weights, dict):
            radii = [weights[i] for i in chem_species]
        elif isinstance(weights, list):
            radii = weights

        conditions = [(dfpoints["Atom"] == i) for i in chem_species]
        choice_weight = [i**2 for i in radii]

        dfpoints["w"] = np.select(conditions, choice_weight)
        dfpoints["x"] = pd.to_numeric(dfpoints["x"])
        dfpoints["y"] = pd.to_numeric(dfpoints["y"])
        dfpoints["z"] = pd.to_numeric(dfpoints["z"])

        points = dfpoints[["x", "y", "z", "w"]].to_numpy()
        simplices = diode.fill_weighted_alpha_shapes(points)
        f = dionysus.Filtration(simplices)
        m = dionysus.homology_persistence(f, progress=True)
        dgms = dionysus.init_diagrams(m, f)

        # Gather the PD of loop in a dataframe
        dfPD = pd.DataFrame(data={"Birth": [p.birth for p in dgms[dimension]],
                                  "Death": [p.death for p in dgms[dimension]],
                                  })
        return dfPD
    
    def get_angular_dist(self, center_atom, neighbor_atoms):
        print('Place_holder')
    



def get_msd():
    print('Place_holder')




xyz_start = io.read(f"/Volumes/My Passport for Mac/sodium silicate/30Na_800/data/1/propensity/1/md.lammpstrj", index=0 , format="lammps-dump-text")
print(xyz_start.get_positions())