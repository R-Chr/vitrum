from vitrum.utility import get_random_packed
from vitrum.batch_active.flows import strained_flows
from itertools import product
from jobflow.managers.fireworks import flow_to_workflow
import numpy as np
import uuid


class balace:
    def __init__(self, lp, units, mp_api_key):
        self.units = units
        self.mp_api_key = mp_api_key
        self.lp = lp
        self.runs = {}

    def gen_even_structures(
        self,
        spacing: int = 10,
    ) -> list:

        lists = [np.int32(np.linspace(0, 100, int(100 / spacing + 1))) for i in range(len(self.units))]
        all_combinations = product(*lists)
        valid_combinations = [combo for combo in all_combinations if sum(combo) == 100]
        structures = []
        for comb in valid_combinations:
            atoms_dict = {str(self.units[0]): comb[0], str(self.units[1]): comb[1], str(self.units[2]): comb[2]}
            structures.append(
                get_random_packed(
                    atoms_dict, target_atoms=100, minAllowDis=1.7, mp_api_key=self.mp_api_key, datatype="pymatgen"
                )
            )
        return structures

    def high_temp_run(self):
        run_id = uuid.uuid4()
        structures = self.gen_even_structures()
        for i in structures:
            flow = strained_flows(structures, self.mp_api_key, metadata=run_id)
            wf = flow_to_workflow(flow)
            self.lp.add_wf(wf)
        self.runs.update({"high_temp_run": run_id})

    # def get_atoms_from_wf(fw_id):
    #     atoms = []
    #     wf = lp.get_wf_summary_dict(fw_id)
    #     for fw in wf['states']:
    #         if wf['states'][fw] == 'COMPLETED':
    #             dirs = wf['launch_dirs'][fw][0]
    #             atoms_fw = read(f'{dirs}/OUTCAR.gz', format='vasp-out',index=":")
    #             atoms = atoms + atoms_fw
    #     return atoms
