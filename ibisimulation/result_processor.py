import os
import numpy as np
from joblib import Parallel, delayed
from scipy.constants import Avogadro as NA
from . import tools_lammps as tl_lmp
from . import tools_structure as tcp

class ResultProcessor:
    def __init__(self, directory, config, initializer, alpha):
        self.directory = directory
        self.config = config
        self.init = initializer
        self.alpha = alpha
        self._alias_initializer_attributes()

    def _alias_initializer_attributes(self):
        for attr in [
            "pdf_ref", "r_RDF_ref", "RDF_cutoff", "RDF_delta_r", "bond_types",
            "r_bond", "r_bond_dist", "bond_length_ref", "angle_types",
            "r_angle", "r_angle_dist", "angle_dist_ref"
        ]:
            setattr(self, attr, getattr(self.init, attr))

    def run(self):
        dump_file = os.path.join(self.directory, "dump.xyz")
        output_file = os.path.join(self.directory, "out.dat")
        log_file = os.path.join(self.directory, "log.lammps")

        frame, _, L_list = tl_lmp.read_lammps_dump_custom(dump_file)
        nframes = len(frame)

        self.lmp = tl_lmp.read_lammps_full(output_file)
        self.lmp.atom_info = self.lmp.atom_info[np.argsort(self.lmp.atom_info[:, 0])]
        self.mass_AA = self.lmp.mass[self.lmp.atom_info[:, 2].astype(int)-1, 1].astype(float)
        self.total_mass = np.sum(self.mass_AA)
        self.type_atom = self.lmp.atom_info[:, 2]

        self._compute_rdf_matrix()

        args = [(self.lmp, frame[i], L_list[i]) for i in np.arange(int(nframes / 2), nframes)]
        results = Parallel(n_jobs=self.config["n_cpus"])(delayed(self._process_frame)(*arg) for arg in args)

        property_avg = self._average_results(results)
        log = tl_lmp.read_log_lammps(log_file)

        if self.config["target"] == "density":
            target_val = np.array(log[-1]["Density"])
        elif self.config["target"] == "pressure":
            target_val = np.array(log[-1]["Press"])
        else:
            raise ValueError("Target not supported")

        property_avg["target"] = np.mean(target_val)
        self.box_size = np.mean(log[-1]["Lx"])

        self._write_output(property_avg)
        return property_avg

    def _compute_rdf_matrix(self):
        self.compute_RDF_matrix = {}
        for i in range(self.lmp.natom_types):
            for j in range(i, self.lmp.natom_types):
                idx_type1 = np.where(self.type_atom == i+1)[0]
                idx_type2 = np.where(self.type_atom == j+1)[0]
                mol_type1 = self.lmp.atom_info[idx_type1, 1]
                mol_type2 = self.lmp.atom_info[idx_type2, 1]
                self.compute_RDF_matrix[(i+1, j+1)] = (mol_type1[:, None] != mol_type2[None, :]).astype(int)

    def _process_frame(self, lmp, frame, L):
        # To be implemented: density, RDF, bond lengths, angles
        return {}

    def _average_results(self, results):
        property_avg = {}
        for result in results:
            for key1 in result:
                if isinstance(result[key1], dict):
                    if key1 not in property_avg:
                        property_avg[key1] = {}
                    for key2 in result[key1]:
                        property_avg[key1].setdefault(key2, []).append(result[key1][key2])
                else:
                    property_avg.setdefault(key1, []).append(result[key1])

        for key1 in property_avg:
            if isinstance(property_avg[key1], dict):
                for key2 in property_avg[key1]:
                    property_avg[key1][key2] = np.mean(property_avg[key1][key2], axis=0)
            else:
                property_avg[key1] = np.mean(property_avg[key1], axis=0)

        return property_avg

    def _write_output(self, property_avg):
        if "RDF" in property_avg:
            g_list = [property_avg["RDF"][key] for key in property_avg["RDF"]]
            np.savetxt(os.path.join(self.directory, "pdf.txt"), np.column_stack((self.r_RDF_ref, np.array(g_list).T)))

        if "bl" in property_avg:
            bl_list = [property_avg["bl"][key] for key in property_avg["bl"]]
            np.savetxt(os.path.join(self.directory, "bond_length_dist.txt"), np.column_stack((self.r_bond, np.array(bl_list).T)))

        if "angle" in property_avg:
            angle_list = [property_avg["angle"][key] for key in property_avg["angle"]]
            np.savetxt(os.path.join(self.directory, "angle_dist.txt"), np.column_stack((self.r_angle, np.array(angle_list).T)))

