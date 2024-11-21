import sys
import os

from experiments_rnn import train_model, eval_model_on_data_config


target_nodes = {"Net1": ['10', '11', '12', '13', '21', '22', '23', '31', '32', '2'],
                "Hanoi": ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                          '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
                          '23', '24', '25', '26', '27', '28', '29', '30', '31', '32']}

results_path = "results"


def run_randomdemand_spike_vs_rest(net_desc: str, target_node_id: str) -> None:
    folder_out = "randomdemand_spike_vs_rest"

    data_configs = [{"random_demands": True, "cl_injection_pattern_desc": "spike"}]
    train_model(net_desc, target_node_id, data_configs,
                dir_out=os.path.join(results_path, net_desc, folder_out))

    test_data_configs = [{"random_demands": False, "cl_injection_pattern_desc": "spike"},
                         {"random_demands": False, "cl_injection_pattern_desc": "wave"},
                         {"random_demands": False, "cl_injection_pattern_desc": "random"},
                         {"random_demands": True, "cl_injection_pattern_desc": "wave"},
                         {"random_demands": True, "cl_injection_pattern_desc": "random"}]
    eval_model_on_data_config(net_desc, target_node_id, test_data_configs,
                              dir_in=os.path.join(results_path, net_desc, folder_out),
                              f_out=os.path.join(results_path, net_desc, folder_out,
                                                 f"rnn_node{target_node_id}.bin"))


def run_spike_vs_rest(net_desc: str, target_node_id: str) -> None:
    folder_out = "spike_vs_rest"

    data_configs = [{"random_demands": False, "cl_injection_pattern_desc": "spike"}]
    train_model(net_desc, target_node_id, data_configs,
                dir_out=os.path.join(results_path, net_desc, folder_out))

    test_data_configs = [{"random_demands": False, "cl_injection_pattern_desc": "wave"},
                         {"random_demands": False, "cl_injection_pattern_desc": "random"},
                         {"random_demands": True, "cl_injection_pattern_desc": "spike"},
                         {"random_demands": True, "cl_injection_pattern_desc": "wave"},
                         {"random_demands": True, "cl_injection_pattern_desc": "random"}]
    eval_model_on_data_config(net_desc, target_node_id, test_data_configs,
                              dir_in=os.path.join(results_path, net_desc, folder_out),
                              f_out=os.path.join(results_path, net_desc, folder_out,
                                                 f"rnn_node{target_node_id}.bin"))


def run_nonrand_vs_rand(net_desc: str, target_node_id: str) -> None:
    folder_out = "nonrand_vs_rand"

    data_configs = [{"random_demands": False, "cl_injection_pattern_desc": "spike"},
                    {"random_demands": False, "cl_injection_pattern_desc": "random"}]
    train_model(net_desc, target_node_id, data_configs,
                dir_out=os.path.join(results_path, net_desc, folder_out))

    test_data_configs = [{"random_demands": False, "cl_injection_pattern_desc": "wave"},
                         {"random_demands": True, "cl_injection_pattern_desc": "spike"},
                         {"random_demands": True, "cl_injection_pattern_desc": "wave"},
                         {"random_demands": True, "cl_injection_pattern_desc": "random"}]
    eval_model_on_data_config(net_desc, target_node_id, test_data_configs,
                              dir_in=os.path.join(results_path, net_desc, folder_out),
                              f_out=os.path.join(results_path, net_desc, folder_out,
                                                 f"rnn_node{target_node_id}.bin"))


def run_rand_vs_all(net_desc: str, target_node_id: str) -> None:
    folder_out = "rand_vs_all"

    data_configs = [{"random_demands": True, "cl_injection_pattern_desc": "spike"},
                    {"random_demands": True, "cl_injection_pattern_desc": "random"}]
    train_model(net_desc, target_node_id, data_configs,
                dir_out=os.path.join(results_path, net_desc, folder_out))

    test_data_configs = [{"random_demands": False, "cl_injection_pattern_desc": "wave"},
                         {"random_demands": False, "cl_injection_pattern_desc": "spike"},
                         {"random_demands": False, "cl_injection_pattern_desc": "random"},
                         {"random_demands": True, "cl_injection_pattern_desc": "wave"}]
    eval_model_on_data_config(net_desc, target_node_id, test_data_configs,
                              dir_in=os.path.join(results_path, net_desc, folder_out),
                              f_out=os.path.join(results_path, net_desc, folder_out,
                                                 f"rnn_node{target_node_id}.bin"))


def run_allin(net_desc: str, target_node_id: str) -> None:
    folder_out = "all-in"

    data_configs = [{"random_demands": False, "cl_injection_pattern_desc": "spike"},
                    {"random_demands": False, "cl_injection_pattern_desc": "random"},
                    {"random_demands": True, "cl_injection_pattern_desc": "spike"},
                    {"random_demands": True, "cl_injection_pattern_desc": "random"}]
    train_model(net_desc, target_node_id, data_configs,
                dir_out=os.path.join(results_path, net_desc, folder_out))

    test_data_configs = [{"random_demands": False, "cl_injection_pattern_desc": "wave"},
                         {"random_demands": True, "cl_injection_pattern_desc": "wave"}]
    eval_model_on_data_config(net_desc, target_node_id, test_data_configs,
                              dir_in=os.path.join(results_path, net_desc, folder_out),
                              f_out=os.path.join(results_path, net_desc, folder_out,
                                                 f"rnn_node{target_node_id}.bin"))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: <net_desc> <target_node_id>")

    net_desc = sys.argv[1]
    target_node_id = target_nodes[net_desc][int(sys.argv[2])-1]
    print(net_desc, target_node_id)

    # Run different configurations
    run_nonrand_vs_rand(net_desc, target_node_id)
    run_rand_vs_all(net_desc, target_node_id)

    run_allin(net_desc, target_node_id)

    run_spike_vs_rest(net_desc, target_node_id)
    run_randomdemand_spike_vs_rest(net_desc, target_node_id)
