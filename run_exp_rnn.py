import sys
import os

from experiments_rnn import train_model, eval_model_on_data_config


target_nodes = {"Net1": ['10', '11', '12', '13', '21', '22', '23', '31', '32', '2'],
                "Hanoi": ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                          '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
                          '23', '24', '25', '26', '27', '28', '29', '30', '31', '32'],
                "CY-DBP": ['dist1', 'dist11', 'dist30', 'dist33', 'dist35', 'dist51', 'dist64',
                           'dist66', 'dist68', 'dist71', 'dist78', 'dist83', 'dist90', 'dist97',
                           'dist100', 'dist103', 'dist105', 'dist106', 'dist110', 'dist115',
                           'dist116', 'dist136', 'dist160', 'dist165', 'dist179', 'dist181',
                           'dist183', 'dist195', 'dist209', 'dist219', 'dist223', 'dist225',
                           'dist231', 'dist273', 'dist275', 'dist284', 'dist288', 'dist289',
                           'dist295', 'dist306', 'dist310', 'dist312', 'dist330', 'dist337',
                           'dist341', 'dist342', 'dist348', 'dist354', 'dist356', 'dist361',
                           'dist376', 'dist379', 'dist383', 'dist394', 'dist398', 'dist399',
                           'dist406', 'dist408', 'dist412', 'dist415', 'dist417', 'dist420',
                           'dist424', 'dist426', 'dist439', 'dist443', 'dist446', 'dist458',
                           'dist470', 'dist473', 'dist485', 'dist490', 'dist494', 'dist504',
                           'dist521', 'dist523', 'dist525', 'dist532', 'dist546', 'dist555',
                           'dist559', 'dist585', 'dist592', 'dist593', 'dist606', 'dist616',
                           'dist622', 'dist628', 'dist631', 'dist632', 'dist648', 'dist650',
                           'dist655', 'dist656', 'dist679', 'dist680', 'dist681', 'dist688',
                           'dist701', 'dist707', 'dist708', 'dist745', 'dist759', 'dist772',
                           'dist787', 'dist789', 'dist806', 'dist819', 'dist830', 'dist839',
                           'dist852', 'dist865', 'dist885', 'dist889', 'dist891', 'dist896',
                           'dist908', 'dist931', 'dist932', 'dist989', 'dist993', 'dist996',
                           'dist998', 'dist999', 'dist1001', 'dist1003', 'dist1020', 'dist1025',
                           'dist1028', 'dist1031', 'dist1039', 'dist1075', 'dist1081', 'dist1126',
                           'dist1130', 'dist1144', 'dist1155', 'dist1165', 'dist1168', 'dist1179',
                           'dist1185', 'dist1196', 'dist1202', 'dist1214', 'dist1225', 'dist1226',
                           'dist1229', 'dist1241', 'dist1245', 'dist1252', 'dist1255', 'dist1268',
                           'dist1272', 'dist1283', 'dist1284', 'dist1287', 'dist1292', 'dist1294',
                           'dist1303', 'dist1304', 'dist1308', 'dist1309', 'dist1311', 'dist1313',
                           'dist1332', 'dist1339', 'dist1343', 'dist1345', 'dist1346', 'dist1363',
                           'dist1367', 'dist1369', 'dist1371', 'dist1372', 'dist1406', 'dist1413',
                           'dist1443', 'dist1446', 'dist1454', 'dist1455', 'dist1459', 'dist1464',
                           'dist1468', 'dist1471', 'dist1472', 'dist1496', 'dist1499', 'dist1502',
                           'dist1511', 'dist1522', 'dist1528', 'dist1539', 'dist1550', 'dist1557',
                           'dist1568', 'dist1607', 'dist1608', 'dist1611', 'dist1622', 'dist1634',
                           'dist1636', 'dist1656', 'dist1687', 'dist1689', 'dist1694', 'dist1702',
                           'dist1722', 'dist1723', 'dist1740', 'dist1741', 'dist1765', 'dist1768',
                           'dist1769', 'dist1770', 'dist1786', 'dist1788', 'dist1801', 'dist1802',
                           'dist1808', 'dist1812', 'dist1829', 'dist1840', 'dist1843', 'dist1850',
                           'dist1854', 'dist1869', 'dist1876', 'dist1877', 'dist1878', 'dist1887',
                           'dist1892', 'dist1893', 'dist1900', 'dist1901', 'dist1903', 'dist1906',
                           'dist1915', 'dist1924', 'dist1931', 'dist1936', 'dist1944', 'dist1962',
                           'dist1973', 'dist1975', 'dist1978', 'dist1979', 'dist1984', 'dist1987',
                           'dist1988', 'dist412_new', '240', '292', '296', '348', '350', '406', '408',
                           '414', '419', '423', '425', '426', '427', '519', '520', '542', '543', '544',
                           '545', '546', '555', 'N13', 'N14', 'N15', 'N17', 'N18', '240-1', 'N15_1',
                           '415', '410', '293', '1', '2', '3', 'WTP', 'Desalination']
                }
results_path = ""


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
    if len(sys.argv) != 4:
        raise ValueError("Usage: <net_desc> <target_node_idx> <config>")

    net_desc = sys.argv[1]
    target_node_id = target_nodes[net_desc][int(sys.argv[2])-1]
    config = int(sys.argv[3])
    print(net_desc, target_node_id)

    # Run different configurations
    if config == 0:
        run_nonrand_vs_rand(net_desc, target_node_id)
    elif config == 1:
        run_rand_vs_all(net_desc, target_node_id)
    elif config == 2:
        run_allin(net_desc, target_node_id)
    elif config == 3:
        run_spike_vs_rest(net_desc, target_node_id)
    elif config == 4:
        run_randomdemand_spike_vs_rest(net_desc, target_node_id)
