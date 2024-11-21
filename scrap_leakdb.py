import os
import shutil
from epyt_flow.simulation import ScenarioSimulator


if __name__ == "__main__":
    # Hanoi
    path_to_leakdb = "data/LeakDB/Hanoi_CMH/"
    path_out = "data/Networks/Hanoi"

    for s_id in range(1, 1001):
        f_inp_in = os.path.join(path_to_leakdb, f"Scenario-{s_id}",
                                f"Hanoi_CMH_Scenario-{s_id}.inp")
        shutil.copyfile(f_inp_in, os.path.join(path_out, f"Scenario-{s_id}.inp"))

    with ScenarioSimulator(f_inp_in="data/Networks/Hanoi/Scenario-1.inp") as sim:
        sim.get_topology().save_to_file("data/Networks/Hanoi/topology")

    # Net1
    path_to_leakdb = "data/LeakDB/Net1_CMH/"
    path_out = "data/Networks/Net1"

    for s_id in range(1, 1001):
        f_inp_in = os.path.join(path_to_leakdb, f"Scenario-{s_id}",
                                f"Net1_CMH_Scenario-{s_id}.inp")
        shutil.copyfile(f_inp_in, os.path.join(path_out, f"Scenario-{s_id}.inp"))

    with ScenarioSimulator(f_inp_in="data/Networks/Net1/Scenario-1.inp") as sim:
        sim.get_topology().save_to_file("data/Networks/Net1/topology")
