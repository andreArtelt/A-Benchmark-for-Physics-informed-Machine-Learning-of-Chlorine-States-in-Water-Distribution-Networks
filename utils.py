import os
import numpy as np
from epyt_flow.simulation import ScenarioSimulator
import matplotlib.pyplot as plt

from data_loader import DataLoader
import networkx as nx


def visualize_error_on_topology(net_desc: str, scores_per_node: dict[str, float], show: bool = True) -> None:
    f_inp_in = "data/Networks/Hanoi/Scenario-1.inp" if net_desc == "Hanoi" else "data/Networks/Net1/Scenario-1.inp"
    with ScenarioSimulator(f_inp_in=f_inp_in) as s:
        all_nodes = s.epanet_api.getNodeNameID()

        node_values = []
        for node_id in all_nodes:
            if node_id in scores_per_node:
                node_values.append(scores_per_node[node_id])
            else:
                reservoir_node_id = node_id
                node_values.append(0)

        s.epanet_api.plot(node_values=node_values,
                          highlightnode=reservoir_node_id,
                          figure=False)

        if show is True:
            plt.show()


def merge_data(data: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Merges tuples/pairs of input and output data into a single tuple/pair of input and output data.

    Parameters
    ----------
    data : `list[tuple[numpy.ndarray, numpy.ndarray]]`
        List of tuples/pairs of input and output data.

    Returns
    -------
    `tuple[numpy.ndarray, numpy.ndarray]`
        Tuple/Pair of input and output data.
    """
    X, y = [], []

    for X_, y_ in data:
        X.append(X_)
        y.append(y_)

    X = np.concatenate(X)
    y = np.concatenate(y)

    return X, y


def estimate_transport_delay(net_desc: str, path_to_data: str = "data") -> dict:
    """
    Estimates the transport delay for each node -- i.e. time the water needs from
    the chlorine injection point to every other node

    Parameters
    ----------
    net_desc : `str`
        Name of the network. Must be either "Net1" or "Hanoi".
    path_to_data : `str`, optional
        Path to the data folder.
        The default is "data"

    Returns
    -------
    `dict`
        Transport delay for each node (node ID is mapped to transport delay).
    """
    # Load data
    f_in = os.path.join(path_to_data, "chlorine-data", net_desc,
                        "randomized_demands=True-spike", "0.npz")
    _, chlorine_data, _ = DataLoader(path_to_data).load_unprocessed_data_from_file(f_in)

    # Load network topology
    topo = DataLoader(path_to_data).load_network_topology(net_desc)

    # Compute/Estimate the time the water needs from the chlorine injection point to every
    # other node -- this time is called "transport delay"
    nodes_dist_time = []
    for i in range(chlorine_data.shape[1]):
        nodes_dist_time.append(np.argwhere(chlorine_data[:, i] > 0.001)[0][0])

    return dict(zip(topo.nodes, nodes_dist_time))


def plot_graph_from_topology(topology, node_colors='#96a6d4', edge_colors='#2c436d',
                             show_colorbar=False, with_labels=True, relabel_nodes=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))

    G = nx.Graph(topology)
    vmin, vmax = None, None
    if isinstance(node_colors, dict):
        all_colors = list(node_colors.values())
        if not isinstance(all_colors[0], str):
            vmin = min(list(node_colors.values()))
            vmax = max(list(node_colors.values()))

    node_map = dict(zip(G.nodes, range(len(G))))
    node_map_inv = dict(zip(range(len(G)), G.nodes))
    pos = { node_map[n] : topology.get_node_info(n)['coord'] for n in topology }
    # pos = { n : topology.get_node_info(n)['coord'] for n in topology }
    G = nx.relabel_nodes(G, node_map)
    
    pipes_plot = nx.draw_networkx_edges(
        G, pos=pos, width=3, edge_color=edge_colors, ax=ax, node_size=380,
        #edgelist=[ (u,v) for u, v, d in G.edges(data=True) if d['info']['type'] == 'PIPE' ]
    )
    pipes_plot.set_label('Pipe')
    pipes_plots = [pipes_plot]
    pumps = [ (v,u) for u, v, d in G.edges(data=True) if d['info']['type'] == 'PUMP' ]
    for pu, pv in pumps:
        pumps_plot = nx.draw_networkx_nodes(
            nx.Graph([[0,0]]), pos={ 0 : np.add(pos[pu], pos[pv])/2 }, node_color=edge_colors, ax=ax, node_size=150,
            node_shape='D'
        )
    if len(pumps):
        pumps_plot.set_label('Pump')
        pipes_plot = pipes_plots.append(pumps_plot)
    node_types = [ n[1]['info']['type'] for n in topology.nodes(data=True) ]
    node_plots = []
    for marker, node_type in zip('ovs', ['JUNCTION', 'TANK', 'RESERVOIR']):
        subset = [ node_map[n[0]] for n in topology.nodes(data=True) if n[1]['info']['type'] == node_type ]
        # subset = [ n[0] for n in topology.nodes(data=True) if n[1]['info']['type'] == node_type ]
        if isinstance(node_colors, dict):
            nc = [ node_colors.get(node_map_inv[n], 0) for n in subset ]
        else:
            nc = node_colors
        node_plot = nx.draw_networkx_nodes(
            nx.subgraph(G, subset), pos=pos, node_color=nc, ax=ax, node_size=380, node_shape=marker,
            vmin=vmin, vmax=vmax
        )
        if with_labels:
            nx.draw_networkx_labels(
                nx.subgraph(G, subset), pos=pos, ax=ax, #labels=dict(zip(G.nodes, node_map.keys()))
                labels=node_map_inv if not relabel_nodes else None
            )
        node_plot.set_label(node_type.capitalize())
        node_plots.append(node_plot)
    ax.axis('off')
    os.makedirs('Figures', exist_ok=True)
    if show_colorbar:
        plt.gcf().colorbar(node_plots[0], ax=ax)

    return [ *pipes_plots, *node_plots, ]