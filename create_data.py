import os
import random
from pathlib import Path
import numpy as np
from epyt_flow.simulation import ScenarioSimulator, ToolkitConstants
from epyt_flow.utils import to_seconds


def create_spike_pattern(pattern_length: int) -> np.ndarray:
    steps = np.array([*range(1, pattern_length-1, 1)])
    pattern_mult = .5*np.sin(steps*.5) + .5

    pattern_mult[8] = 0.0001
    spike_pattern = np.copy(pattern_mult[:9])
    pattern_mult[9:] = 0

    delay = 50  # Spike pattern - "impulse-response"
    rand_offset = 50
    cur_idx = 9

    pattern_len = len(pattern_mult)
    len_pattern = len(spike_pattern)
    while (cur_idx + delay + rand_offset + len_pattern) < pattern_len:
        cur_idx += delay + random.randint(0, rand_offset)
        pattern_mult[cur_idx:cur_idx+len_pattern] = np.copy(spike_pattern)
        cur_idx += len_pattern

    return pattern_mult


def create_random_pattern(pattern_length: int) -> np.ndarray:
    steps = np.array([*range(1, pattern_length-1, 1)])
    pattern_mult = .5*np.sin(steps*.5) + .5

    pattern_mult = np.random.rand(pattern_mult.shape[0])

    return pattern_mult


def create_wave_pattern(pattern_length: int) -> np.ndarray:
    steps = np.array([*range(1, pattern_length-1, 1)])

    # Randomize: magnitude * (sin(x * width) + 1)
    m = np.random.rand()
    w = np.random.rand()
    pattern_mult = m * (np.sin(steps * w) + 1)

    return pattern_mult


def run_sim(f_inp_in: str, f_out, pattern_type: str = "spike",
            randomized_demands: bool = False) -> None:
    with ScenarioSimulator(f_inp_in=f_inp_in) as sim:
        # Set general parameters
        sim.set_general_parameters(simulation_duration=to_seconds(days=30),
                                   hydraulic_time_step=1800,
                                   quality_time_step=300)

        # Set initial concentration and simple (constant) reactions
        zeroNodes = [0] * sim.epanet_api.getNodeCount()
        sim.epanet_api.setNodeInitialQuality(zeroNodes)
        sim.epanet_api.setLinkBulkReactionCoeff([-.5] * sim.epanet_api.getLinkCount())
        sim.epanet_api.setLinkWallReactionCoeff([-.01] * sim.epanet_api.getLinkCount())

        # Add chlorine injection at the reservoir
        sim.enable_chemical_analysis()

        pattern_length = max(sim.epanet_api.getPatternLengths())
        if pattern_type == "spike":
            pattern = create_spike_pattern(pattern_length)
        elif pattern_type == "wave":
            pattern = create_wave_pattern(pattern_length)
        elif pattern_type == "random":
            pattern = create_random_pattern(pattern_length)
        else:
            raise ValueError("Unknown pattern type")

        reservoir_id = sim.epanet_api.getNodeReservoirNameID()[0]
        sim.add_quality_source(node_id=reservoir_id,
                               pattern=pattern,
                               source_type=ToolkitConstants.EN_CONCEN)

        # Place quality and flow sensor everywhere
        sim.set_flow_sensors(sensor_locations=sim.sensor_config.links)
        sim.set_node_quality_sensors(sensor_locations=sim.sensor_config.nodes)
        sim.set_link_quality_sensors(sensor_locations=sim.sensor_config.links)

        # Randomizes all demands if requested
        if randomized_demands is True:
            sim.randomize_demands()

        # Run simulation and store results
        res = sim.run_simulation()

        np.savez(f_out,
                 injection_node_id=reservoir_id,
                 injection_pattern=pattern,
                 injection_node_idx=sim.sensor_config.nodes.index(reservoir_id),
                 node_ids=sim.sensor_config.nodes,
                 link_ids=sim.sensor_config.links,
                 flow_data=res.get_data_flows(),
                 node_quality=res.get_data_nodes_quality(),
                 link_chlorine_data=res.get_data_links_quality())


def create_dataset(network_desc="Hanoi"):
    # Specifies path to scenarios and chlorine injection pattern type
    path_to_scenarios = os.path.join("data", "Networks", network_desc)

    for randomized_demands in [True, False]:
        for pattern_type in ["spike", "wave", "random"]:
            # Simulate all scenarios
            scenarios = [os.path.join(path_to_scenarios,
                                      f"Scenario-{i}.inp") for i in range(1, 1001)]
            f_out_path = os.path.join("data", "chlorine-data", network_desc,
                                      f"randomized_demands={randomized_demands}-{pattern_type}")
            Path(f_out_path).mkdir(parents=True, exist_ok=True)

            for f_inp_in, i in zip(scenarios, range(len(scenarios))):
                print(i, f_inp_in)
                run_sim(f_inp_in, os.path.join(f_out_path, f"{i}.npz"), pattern_type,
                        randomized_demands)


if __name__ == "__main__":
    create_dataset("Net1")
    create_dataset("Hanoi")