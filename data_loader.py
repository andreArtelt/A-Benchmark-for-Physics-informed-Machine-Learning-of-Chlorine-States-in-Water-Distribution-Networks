import os
import random
import numpy as np
from epyt_flow.simulation import ScenarioSimulator
from epyt_flow.topology import NetworkTopology
from multiprocessing import Pool
from itertools import repeat


class DataLoader:
    """
    Class for loading the chlorine state estimation benchmark data set.

    Parameters
    ----------
    path_to_data : `str`, optional
        Path to the data folder.
        The default is "data"
    """
    def __init__(self, path_to_data: str = "data"):
        self.__path_to_data = path_to_data
        self._path_to_chlorine_data = os.path.join(self.__path_to_data, "chlorine-data")

    def load_unprocessed_data_from_file(self, f_in: str) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Loads the unprocessed data from a given scenario file (.npz).

        Paramaters
        ----------
        f_in : `str`
            Scenario file (.npz) containing the data.

        Returns
        -------
        `tuple[numpy.ndarray, numpy.ndarray, int]`
            Tuple of flow data, Chlorine concentrations (at every node),
            ID of node at which Chlorine is injected.
        """
        data = np.load(f_in)
        flow_data, chlorine_data, injection_nodes_idx = data["flow_data"], data["node_quality"], \
            data["injection_node_idx"]

        return flow_data, chlorine_data, [int(n_id) for n_id in injection_nodes_idx]

    def load_unprocessed_data_with_topology(self, f_in, f_inp_in, path_in):
        X_flows, X_cl_conc, injection_nodes_idx = self.load_unprocessed_data_from_file(os.path.join(path_in, f_in))

        with ScenarioSimulator(f_inp_in=f_inp_in) as s:
            topo = s.get_topology()

        return X_flows, X_cl_conc, topo, injection_nodes_idx

    def load_unprocessed_data_from_scenarios(self, net_desc: str, random_demands: bool,
                                             cl_injection_pattern_desc: str
                                             ) -> tuple[list[np.ndarray], list[np.ndarray],
                                                        int, NetworkTopology]:
        """
        Loads the unprocessed data from given scenarios.

        Parameters
        ----------
        net_desc : `str`
            Name of the network. Must be either "Net1" or "Hanoi".
        random_demands : `bool`
            True if scenarios with randomized demands are requested, False otherwise.
        cl_injection_pattern_desc : `str`
            Name of the Chlorine injection pattern. Must be one of the following:
                - "spike"
                - "random"
                - "wave"

        Returns
        -------
        `tuple[list[numpy.ndarray], list[numpy.ndarray], int, NetworkTopology]`
            Tuple of flow data (list), Chlorine concentrations at every node (list),
            ID of node at which Chlorine is injected --
            each entry in the list referes to a single scenario..
        """
        if not net_desc in ["Hanoi", "Net1"]:
            raise ValueError("'net_desc' must be either 'Hanoi' or 'Net1'")
        if not cl_injection_pattern_desc in ["spike", "random", "wave"]:
            raise ValueError("'cl_injection_pattern_desc' must be one of the following: " +
                             "'spike', 'random', 'wave'")

        path_in = os.path.join(self._path_to_chlorine_data, net_desc,
                               f"randomized_demands={random_demands}-{cl_injection_pattern_desc}")
        files_in = [f for f in os.listdir(path_in)
                    if os.path.isfile(os.path.join(path_in, f)) and f.endswith(".npz")]
        inp_files_in = [os.path.join(self.__path_to_data, "Networks", net_desc,
                                     f"Scenario-{int(file_name.replace('.npz', ''))+1}.inp") for file_name in files_in]

        X_flows, X_cl_conc, topos = [], [], []
        injection_nodes_idx = None
        ncpus = 20 # os.cpu_count()
        with Pool(processes=ncpus) as pool:
            jobs = pool.starmap(
                self.load_unprocessed_data_with_topology, 
                zip(files_in, inp_files_in, repeat(path_in)), 
                chunksize=len(files_in)//ncpus
            )
            for X_flows_, X_cl_conc_, topo, injection_nodes_idx in jobs:
                X_flows.append(X_flows_)
                X_cl_conc.append(X_cl_conc_)
                topos.append(topo)

        return np.array(X_flows), np.array(X_cl_conc), injection_nodes_idx, topos

    def _prepare_data(self, flows: np.ndarray, chlorine: np.ndarray, target_node_idx: int,
                      injection_nodes_idx: int) -> tuple[np.ndarray, np.ndarray]:
        X = []
        y = []

        cur_idx = 0
        while cur_idx < flows.shape[0]:
            X.append(np.concatenate((flows[cur_idx, :].flatten(),
                                     chlorine[cur_idx, injection_nodes_idx].flatten())))
            y.append(chlorine[cur_idx, target_node_idx])    # Predict current Cl concentration

            cur_idx += 1

        return np.array(X), np.array(y)

    def load_data_from_file(self, f_in: str, target_node_id: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads and process data from a given scenario file (.npz).

        Parameters
        ----------
        f_in : `str`
            Scenario file (.npz) containing the data.
        target_node_id : `str`
            ID of the node for which the Chlorine concentration is to be estimated.

        Returns
        -------
        `tuple[numpy.ndarray, numpy.ndarray]`
            Tuple of input data and target values.
        """
        data = np.load(f_in)
        node_ids = data["node_ids"].tolist()
        flow_data, chlorine_data = data["flow_data"], data["node_quality"]

        injection_nodes_id = data["injection_node_id"]
        try:
            [n_id for n_id in injection_nodes_id]
        except:  # Support older versions of the data set
            injection_nodes_id = [injection_nodes_id]
        injection_nodes_id = [n_id for n_id in injection_nodes_id]

        injection_nodes_idx = [node_ids.index(n_id) for n_id in injection_nodes_id]
        X, y = self._prepare_data(flow_data, chlorine_data, node_ids.index(target_node_id),
                                  injection_nodes_idx)

        return X, y

    def load_data_from_scenarios(self, net_desc: str, random_demands: bool,
                                 cl_injection_pattern_desc: str, target_node_id: str
                                 ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Loads and process all data from given scenarios.

        Parameters
        ----------
        net_desc : `str`
            Name of the network. Must be either "Net1" or "Hanoi".
        random_demands : `bool`
            True if scenarios with randomized demands are requested, False otherwise.
        cl_injection_pattern_desc : `str`
            Name of the Chlorine injection pattern. Must be one of the following:
                - "spike"
                - "random"
                - "wave"
        target_node_id : `str`
            ID of the node for which the Chlorine concentration is to be estimated.

        Returns
        -------
        `tuple[list[numpy.ndarray], list[numpy.ndarray]]`
            Tuple of input data (list) and target values (list) --
            each entry in the list referes to a single scenario.
        """
        if not net_desc in ["Hanoi", "Net1", "CY-DBP"]:
            raise ValueError("'net_desc' must be either 'Hanoi', 'Net1', or 'CY-DBP'")
        if not cl_injection_pattern_desc in ["spike", "random", "wave"]:
            raise ValueError("'cl_injection_pattern_desc' must be one of the following: " +
                             "'spike', 'random', 'wave'")

        path_in = os.path.join(self._path_to_chlorine_data, net_desc,
                               f"randomized_demands={random_demands}-{cl_injection_pattern_desc}")
        files_in = [os.path.join(path_in, f) for f in os.listdir(path_in)
                    if os.path.isfile(os.path.join(path_in, f)) and f.endswith(".npz")]

        X, y = [], []
        for f_in in files_in:
            X_, y_ = self.load_data_from_file(f_in, target_node_id)
            X.append(X_)
            y.append(y_)

        return X, y

    def load_data(self, train_size: int, val_size: int, net_desc: str, random_demands: bool,
                  cl_injection_pattern_desc: str, shuffle: bool,
                  target_node_id: str) -> tuple[tuple[np.ndarray, np.ndarray],
                                                tuple[np.ndarray, np.ndarray],
                                                tuple[np.ndarray, np.ndarray]]:
        """
        Loads and process all data from given scenarios --
        also splits data into train, validation, and test set.

        Note that the size of the test set is determined automatically based on
        the size of train and validation set.

        Parameters
        ----------
        train_size : `int`
            Number of training scenarios.
        val_size : `int`
            Number of validation scenarios.
        net_desc : `str`
            Name of the network. Must be either "Net1" or "Hanoi".
        random_demands : `bool`
            True if scenarios with randomized demands are requested, False otherwise.
        cl_injection_pattern_desc : `str`
            Name of the Chlorine injection pattern. Must be one of the following:
                - "spike"
                - "random"
                - "wave"
        shuffle : `bool`
            If True, data set will be shuffled before split into train, validation, and test set.
        target_node_id : `str`
            ID of the node for which the Chlorine concentration is to be estimated.            

        Returns
        -------
        `tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]`
            Training, validation, and test data set -- each set is a tuple/pair
            of input and output dat.
        """
        X, y = self.load_data_from_scenarios(net_desc=net_desc, random_demands=random_demands,
                                             cl_injection_pattern_desc=cl_injection_pattern_desc,
                                             target_node_id=target_node_id)

        X = np.array(X)
        y = np.array(y)

        indices = list(range(0, len(y)))
        if shuffle is True:
            random.shuffle(indices)

        X_train, y_train = X[indices[:train_size]], y[indices[:train_size]]
        X_val, y_val = X[indices[train_size:(train_size + val_size)]],\
            y[indices[train_size:(train_size + val_size)]]
        X_test, y_test = X[indices[(train_size + val_size):]], y[indices[(train_size + val_size):]]

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def load_network_topology(self, net_desc: str) -> NetworkTopology:
        """
        Loads and returns the topology of a given network.

        Parameters
        ----------
        net_desc : `str`
            Name of the network. Must be either "Net1" or "Hanoi".

        Returns
        -------
        `epyt_flow.topology.NetworkTopology`
            Topology of network.
        """
        if not net_desc in ["Hanoi", "Net1", "CY-DBP"]:
            raise ValueError("'net_desc' must be either 'Hanoi', 'Net1', or 'CY-DBP'")

        return NetworkTopology.load_from_file(os.path.join(self.__path_to_data, "Networks",
                                                           net_desc, "topology.epytflow_topology"))
