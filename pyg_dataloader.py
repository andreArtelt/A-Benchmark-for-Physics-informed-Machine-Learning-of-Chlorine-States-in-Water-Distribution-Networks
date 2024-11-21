from torch_geometric.data import Dataset, Data
from data_loader import DataLoader
from collections import defaultdict
from pathlib import Path
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
import torch
import os

TRAIN_FRACTION = 0.7
VALIDATION_FRAC = 0.1
TEST_FRAC = 0.2

assert (TRAIN_FRACTION + VALIDATION_FRAC + TEST_FRAC) == 1

class ChlorineDataset(DataLoader, Dataset):
    """
    A simple torch-geometric water quality dataset. The graph is a water 
    distribution system (WDS) where nodes correspond to consumers and water 
    reservoirs, and edges correspond to the pipes of the system. Chlorine is 
    injected at ChlorineDataset.injection_node_idx and distributes according to 
    water flow (advection) and diffusion. Advection is the dominating
    factor.
    
    Parameters
    ----------
    wdn : `str`
        The name of the WDN to load, either 'Hanoi' or 'Net1'
    path_to_data : `str`, optional
        Path to the data folder.
        The default is "data"
    timesteps : `int`, optional
        The number of timesteps in each sample. If None, returns all timesteps.
    random_demands : `bool`
        True if scenarios with randomized demands are requested, False otherwise.
    pattern : `str`, optional
        Name of the Chlorine injection pattern. Must be one of the following:
            - "spike"
            - "random" (Default)
            - "wave"
    subset : `str`, optional
        The subset of data to load. Either "train" (Default), "validation", or "test".
    bidirectional : bool, optional
        Whether the graph should be loaded with bidirectional edges (u -> v) or single 
        directed edges (u -> v). Single directed directions are derived from the .inp file
        and don't correspond to flow direction.
    selfloops : bool, optional
        Whether to add selfloops to the graph.
    full_edge_features : bool, optional
        Whether to use flows and lengths as edge features (True) or only use
        a flow direction as edge features (-1 if inflow, 1 if outflow and 0 if flow == 0).
    """

    def __init__(
            self, wdn, path_to_data='data', timesteps=None, random_demands=True, pattern='random', 
            subset="train", bidirectional=True, selfloops=False, full_edge_features=True
        ):
        DataLoader.__init__(self, path_to_data=path_to_data)
        Dataset.__init__(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timesteps = timesteps
        self.subset = subset
        self.random_demands = random_demands
        self.pattern = pattern
        self.wdn = wdn
        self.bidirectional = bidirectional
        # self.append_edge_nodes = append_edge_nodes
        self.selfloops = selfloops
        self.full_edge_features = full_edge_features
        self.n_edge_features = 1 + self.full_edge_features
        self.replace_pump_diameters = True
        self.load()
    
    def load_graph(self, net_desc=None):
        """
        Load the edge index of WDN called net_desc, either 'Hanoi' or 'Net1'
        """
        net_desc = self.wdn if net_desc is None else net_desc
        topology = self.load_network_topology(net_desc)
        G = nx.Graph(topology)
        G = nx.relabel_nodes(G, dict(zip(G.nodes, range(len(G)))))
        
        if self.bidirectional:
            G = G.to_directed()
        
        edge_index = np.stack(G.edges).T
        return topology, edge_index

    def len(self):
        return self._len
    
    def make_subsets(self, total_size):
        self.subset_ids = {}
        random_state = np.random.RandomState(42)
        total_set = set(range(total_size))
        self.subset_ids["train"] = random_state.choice(
            total_size, size=int(total_size*TRAIN_FRACTION), replace=False
        )
        self.subset_ids["valid"] = random_state.choice(
            list(total_set - set(self.subset_ids["train"])),
            size=int(total_size*VALIDATION_FRAC)
        )
        train_valid = set(self.subset_ids["train"]) | set(self.subset_ids["valid"])
        self.subset_ids["test"] = list(total_set - train_valid)
        # TODO: Run then remove
        assert set(self.subset_ids["train"]).isdisjoint(self.subset_ids["valid"])
        assert set(self.subset_ids["train"]).isdisjoint(self.subset_ids["test"])
        assert set(self.subset_ids["valid"]).isdisjoint(self.subset_ids["test"])
    
    def load_graph_features(self):
        self.graph_features = defaultdict(list)
        self.diameters_max = 0
        self.lengths_max = 0

        for topology in tqdm(self.topologies, desc='Loading Topologies ...'):
            # G = nx.Graph(topology)
            # G = nx.relabel_nodes(G, dict(zip(G.nodes, range(len(G)))))

            # if self.bidirectional:
            #     G = G.to_directed()

            # edge_index = np.stack(G.edges).T

            d = dict(zip(topology.get_all_nodes(), range(len(topology))))
            edge_index = np.stack(list(map(
                lambda e: (d[e[0]], d[e[1]]), 
                list(zip(*topology.get_all_links()))[1])
            )).T
            if self.bidirectional:
                edge_index = np.concatenate((edge_index, edge_index[::-1]), axis=1)

            lengths = np.stack([
                topology.get_edge_data(u, v)['length'] 
                for _, (u, v) in topology.get_all_links() 
            ])
            link_types = np.stack([
                topology.get_edge_data(u, v)['info']['type']
                for _, (u, v) in topology.get_all_links()
            ])
            diameters = np.stack([
                topology.get_edge_data(u, v)['info']['diameter'] 
                for _, (u, v) in topology.get_all_links()
            ])

            if self.replace_pump_diameters:
                diameters[link_types == 'PUMP'] = diameters.max()
            
            if self.bidirectional:
                lengths = np.concatenate([lengths, lengths], axis=-1)
                diameters = np.concatenate([diameters, diameters], axis=-1)

            # if self.append_edge_nodes: # self.edges_as_nodes:
            #     self.num_edges = len(self.edge_index[0])
            #     edge_node_ids = self.num_nodes + np.arange(self.num_edges)
            #     snd = self.edge_index[0]
            #     rec = self.edge_index[1]
            #     self.edge_index = np.concatenate((
            #         np.stack((snd, edge_node_ids)),
            #         np.stack((edge_node_ids, rec))
            #     ), axis=1)
            #     self.quality_data = np.pad(self.quality_data, ((0,0),(0,0),(0,self.num_edges)))
            #     lengths = np.concatenate((lengths, lengths), axis=-1) / 2
            #     diameters = np.concatenate((diameters, diameters), axis=-1)
            #     self.flow_data = np.concatenate((self.flow_data, self.flow_data), axis=-1)
            #     self.num_nodes = self.num_nodes + self.num_edges
            
            if self.selfloops:
                lengths = np.pad(lengths, ((0, self.num_nodes)))
                diameters = np.pad(diameters, ((0, self.num_nodes)))
                sl_edges = np.stack((np.arange(self.num_nodes), np.arange(self.num_nodes)))
                edge_index = np.concatenate((edge_index, sl_edges), axis=1)
                
            self.graph_features['lengths'].append(lengths)
            self.graph_features['diameters'].append(diameters)
            self.graph_features['edge_indices'].append(edge_index)
            self.lengths_max = max(self.lengths_max, lengths.max())
            self.diameters_max = max(self.diameters_max, diameters.max())
        return self.graph_features

    def load(self):
        """
        Load the data into memory. Prepare some useful variables.
        """
        self.n_scenarios = 0
        self._len = 0
        # self.topology, self.edge_index = self.load_graph()
        # load scenario data
        self.flow_data, self.quality_data, self.injection_node_idx, self.topologies = (
            self.load_unprocessed_data_from_scenarios(
                self.wdn, 
                random_demands=self.random_demands, 
                cl_injection_pattern_desc=self.pattern
            )
        )
        self.flow_data = np.stack(self.flow_data)
        self.quality_data = np.stack(self.quality_data)
        self.make_subsets(len(self.quality_data))
        subset_idx = self.subset_ids[self.subset]
        self.flow_data = self.flow_data[subset_idx]
        self.quality_data = self.quality_data[subset_idx]
        self.num_nodes = self.quality_data.shape[-1]
        self.eval_nodes = torch.arange(self.num_nodes)
        if self.timesteps is None:
            self.timesteps = len(self.flow_data[0])
        self._len = sum(map(
            lambda d: len(d) - self.timesteps + 1, self.flow_data
        ))
        self.graph_features = self.load_graph_features()
        if self.bidirectional:
            self.flow_data = np.concatenate([self.flow_data, -self.flow_data], axis=-1)
        if self.selfloops:
            self.flow_data = np.pad(self.flow_data, ((0,0),(0,0),(0,self.num_nodes)))
        self.steps_per_sample = self.quality_data.shape[-2]
        self.n_scenarios = len(self.quality_data)
        self.quality_max = self.quality_data.max()
        self.quality_mean = self.quality_data.mean()
        self.quality_std = self.quality_data.std()
        self.flow_max = self.flow_data.max()
        self.boundary_mask = torch.eye(len(self.eval_nodes), dtype=bool)[self.injection_node_idx]
        # self.edge_index = torch.tensor(self.edge_index)

    def get_attribute_maxima(self):
        return self.quality_max, self.flow_max, self.diameters_max, self.lengths_max, self.quality_mean, self.quality_std
    
    def set_attribute_maxima(self, quality_max, flow_max, diameters_max, lengths_max, quality_mean, quality_std):
        self.quality_max = quality_max
        self.flow_max = flow_max
        self.diameters_max = diameters_max
        self.lengths_max = lengths_max
        self.quality_mean = quality_mean
        self.quality_std = quality_std
           
    def make_edge_features_full_info(self, scenario_idx, flows):
        lengths = self.graph_features['lengths'][scenario_idx]
        diameters = self.graph_features['diameters'][scenario_idx]
        lengths = np.broadcast_to(lengths[:, None], flows.shape)
        diameters = np.broadcast_to(diameters[:, None], flows.shape)
        flow_ = (flows / self.flow_max)#(60*60))
        dias = diameters / self.diameters_max # (10 * 100)
        crosssection = (dias / 2)**2 * np.pi
        flow_velocities = flow_ / crosssection
        flow_velocities = np.nan_to_num(flow_velocities)
        edge_features = np.stack((flow_velocities, lengths / self.lengths_max), axis=-1)
        return edge_features, self.graph_features['edge_indices'][scenario_idx]

    def make_edge_features_flow_direction_only(self, scenario_idx, flows):
        return np.sign(flows)[..., None], self.graph_features['edge_indices'][scenario_idx]
    
    def get(self, idx):
        # The dataset consists of multiple scenarios. Find the scenario idx first.
        scenario_idx = idx // (self.steps_per_sample - self.timesteps + 1)
        # Find the time idx within a scenario.
        time_idx = idx % (self.steps_per_sample - self.timesteps + 1)
        t_end_input = time_idx + self.timesteps
        node_targets = self.quality_data[scenario_idx][time_idx:t_end_input].T
        flows = self.flow_data[scenario_idx][time_idx:t_end_input].transpose(1,0)
        if self.full_edge_features:
            edge_features, edge_index = self.make_edge_features_full_info(scenario_idx, flows)
        else:
            edge_features, edge_index = self.make_edge_features_flow_direction_only(scenario_idx, flows)
        # mask nodes except injection nodes
        node_features = node_targets
        node_targets = node_targets[self.eval_nodes]
        node_features = torch.tensor(node_features / self.quality_max).float()
        node_targets = torch.tensor(node_targets / self.quality_max).float()
        edge_features = torch.tensor(edge_features).float()
        edge_index = torch.tensor(edge_index).long()
        return Data(
            x=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
            y=node_targets,
            time=time_idx,
            num_nodes=self.num_nodes,
            boundary_index=self.injection_node_idx,
            boundary_mask=self.boundary_mask,
            eval_index=self.eval_nodes
        ).to(self.device)