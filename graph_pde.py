from functools import partial
import torch
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import os, json
from evaluation import Evaluator

RESULTS_DIR = os.path.join('Results', 'Graph-PDE')
DATA_DIR = 'data'
    
def reservoir_boundary_condition(time, x, edge_features, boundary_values, edge_values, boundary_index, dt, T, f):
    # inplace manipulation of the current solution
    t = time
    t_lo = torch.floor(t).int().clip(None, T)
    t_hi = torch.ceil(t).int().clip(None, T)
    boundary = boundary_values[:, t_lo] + (time % 1) * (boundary_values[:, t_hi] - boundary_values[:, t_lo])
    edge_features = edge_values[:, t_lo] + (time % 1) * (edge_values[:, t_hi] - edge_values[:, t_lo])
    x = x.clone()
    x[boundary_index] = f(boundary[boundary_index, None])
    return x, edge_features

def train(dataset, test_dataset, model, epochs, optimizer, loss_fn, dt, T):
    losses, validation_losses = [], []
    epochs_iter = tqdm(range(epochs), desc=f'Training for {epochs} epochs', ncols=80, dynamic_ncols=True)
    maxtime = T
    model.set_eval_times(T)

    for e in epochs_iter:
        
        model.set_eval_times(T)
        for data in tqdm(dataset, leave=False):
            optimizer.zero_grad()
            boundary_condition = partial(
                reservoir_boundary_condition,
                boundary_values=data.x[...,:maxtime+1],
                boundary_index=data.boundary_index,#[...,:maxtime],
                edge_values=data.edge_features[:,:maxtime+1],
                dt=dt, T=T, f=model.m1
            )
            predictions = model(data.x[...,:maxtime], data.edge_index, edge_weight=None, boundary_condition=boundary_condition, boundary_index=data.boundary_index)
            predictions = predictions[data.eval_index]
            loss = loss_fn(
                predictions[~data.boundary_mask,:], 
                data.y[~data.boundary_mask,:maxtime+1]
            )
            
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
            epochs_iter.set_description(f'Loss {losses[-1]:.4f} | Grad: {gn:.2f}')
        
        validation_losses.append(evaluate(test_dataset, model, metrics=[loss_fn], dt=dt, T=T)[1])
        
    return losses, validation_losses

# def train(dataset, test_dataset, model, epochs, optimizer, loss_fn, dt, T):
#     losses, validation_losses = [], []
#     epochs_iter = tqdm(range(epochs), desc=f'Training for {epochs} epochs', ncols=80, dynamic_ncols=True)
#     maxtime = T

#     for e in epochs_iter:
        
#         model.set_eval_times(maxtime)
#         for data in tqdm(dataset, leave=False):
#             samples = np.random.choice(len(data.x) - maxtime - 1, size=2, replace=False)
#             for sample_idx in samples:
#                 optimizer.zero_grad()
#                 _slice = slice(sample_idx, sample_idx + maxtime)
#                 boundary_slice = slice(sample_idx, sample_idx + maxtime + 1)
#                 boundary_condition = partial(
#                     reservoir_boundary_condition,
#                     boundary_values=data.x[...,boundary_slice],
#                     boundary_index=data.boundary_index,#[...,boundary_slice],
#                     edge_values=data.edge_features[:,boundary_slice],
#                     dt=dt, T=T, f=model.m1
#                 )
#                 predictions = model(data.x[...,_slice], data.edge_index, edge_weight=None, boundary_condition=boundary_condition)
#                 predictions = predictions[data.eval_index]

#                 # T-step loss
#                 loss = loss_fn(
#                     predictions[~data.boundary_mask,1:], 
#                     data.y[~data.boundary_mask,1:maxtime+1]
#                 )
                
#                 loss.backward()
#                 gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e10)
#                 optimizer.step()
#                 losses.append(loss.detach().cpu().numpy())
#             epochs_iter.set_description(f'Loss {losses[-1]:.4f} | Grad {gn:.4f}')
        
#         validation_losses.append(evaluate(test_dataset, model, metrics=[loss_fn], dt=dt, T=T)['MSE'])
        
#     return losses, validation_losses

@torch.no_grad
def evaluate(dataset, model, metrics, dt, T):
    maxtime = T - 1#model.config['time']
    model.set_eval_times(maxtime)

    y_preds, y_trues = [], []
    metrics_out = []

    for data in tqdm(dataset, leave=False, desc='Evaluating ...'):
        boundary_condition = partial(
            reservoir_boundary_condition,
            boundary_values=data.x[...,:maxtime+1],
            boundary_index=data.boundary_index,#[...,:maxtime+1],
            edge_values=data.edge_features[:,:maxtime+1],
            dt=dt, T=T, f=model.m1
        )
        predictions = model(data.x[...,:maxtime], data.edge_index, edge_weight=None, boundary_condition=boundary_condition, boundary_index=data.boundary_index)
        predictions = predictions[data.eval_index]
        metrics_out.append(metrics[0](predictions[~data.boundary_mask,1:], data.y[~data.boundary_mask,1:maxtime+1]).detach().cpu().numpy())
        y_preds.append(predictions[~data.boundary_mask,1:].detach().cpu().numpy())
        y_trues.append(data.y[~data.boundary_mask,1:maxtime+1].detach().cpu().numpy())
    
    y_pred = np.concatenate(y_preds, axis=0) * dataset.dataset.datasets[0].quality_max
    y_true = np.concatenate(y_trues, axis=0) * dataset.dataset.datasets[0].quality_max

    results = Evaluator.evaluate_predictions(y_pred, y_true)
    
    return results, np.mean(metrics_out)

if __name__ == '__main__':
    from pyg_dataloader import ChlorineDataset
    from torch_geometric.loader import DataLoader
    from torch.utils.data import ConcatDataset
    from models.pde_function import PDEFunction
    from models.base_model import PDEModel
    import argparse
    from collections import defaultdict
    from joblib import dump, load
    from torcheval.metrics.functional import r2_score, mean_squared_error

    T = 25
    MAX_SEQ_LENGTH = 1441

    cli_parser = argparse.ArgumentParser('GNN-PDE Train/Eval Script')
    cli_parser.add_argument(
        '--train_wdn', type=str, default='Hanoi', help='The WDN used in '\
        'training, choose one from "Hanoi"(Default), "Net1"'
    )
    cli_parser.add_argument(
        '--eval_wdn', type=str, default=None, help='The WDN used in '\
        'evaluation, choose one from "Hanoi", "Net1"; Defaults to --train_wdn'
    )
    cli_parser.add_argument(
        '--train_pattern', type=str, nargs='+', default=['random'], 
        help='The pattern(s) used for training, choose from "spike", "wave", "random"(Default)'
    )
    # cli_parser.add_argument(
    #     '--eval_pattern', type=str, nargs='+', default=['random'], 
    #     help='The pattern(s) used for eval, choose from "spike", "wave", "random"(Default)'
    # )
    # cli_parser.add_argument(
    #     '--eval_demand_mode', type=str, nargs='+', default=['random'], 
    #     help='The demand mode used for training, choose from "normal", "random"(Default)'
    # )
    cli_parser.add_argument(
        '--train_timesteps', '-T', type=int, default=25, 
        help='The number of tinesteps per training sample. Defaults to 25.'
    )
    cli_parser.add_argument(
        '--train_demand_mode', type=str, nargs='+', default=['random'], 
        help='The demand mode used for training, choose from "normal", "random"(Default)'
    )
    # TODO: Add the edge_features argument.
    cli_parser.add_argument(
        '--edge_features', type=str, default='direction', 
        help='The edge features to use, choose from "direction"(Default), "full". '\
             'The prior uses only the flow direction, the latter uses flow value, lendth and diameter.'
    )
    args = cli_parser.parse_args()
    if args.eval_wdn is None:
        args.eval_wdn = args.train_wdn

    config = {
        'time' : args.train_timesteps,
        'hidden_dim' : 128,
        'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'method' : 'euler',
        'step_size' : 1./3.,
        'evaluation_time' : MAX_SEQ_LENGTH,
        'eval_discrete' : True,
        'tol_scale' : 10,
        'input_dim' : 1,
        'output_dim' : 1,
        'n_edge_features' : 2,
        'add_source' : False,
        'dropout' : 0,
        'alpha_train' : True,
        'alpha_sigmoid' : False,
        'learning_rate' : 0.0005,
        'batch_size' : 32,
        'epochs' : 100,
    }

    config.update(vars(args))
    config['n_edge_features'] = 1 + (args.edge_features == 'full') # TODO torch_ds.n_edge_features

    def concat_datasets(datasets, attrib_maxima=None):
        if attrib_maxima is None:
            attrib_maxima = [ ds.get_attribute_maxima() for ds in datasets ]
            attrib_maxima = np.max(attrib_maxima, axis=0)
        for ds in datasets:
            ds.set_attribute_maxima(*attrib_maxima)
        full_ds = ConcatDataset(datasets)
        return full_ds, attrib_maxima

    def make_datasplits(config):
        train_valid_sets = defaultdict(list)
        for split in ['train', 'valid']:
            for train_pattern in config['train_pattern']:
                for demand_mode in config['train_demand_mode']:
                    train_valid_sets[split].append(ChlorineDataset(
                        wdn=config['train_wdn'], pattern=train_pattern, path_to_data=DATA_DIR,
                        timesteps=MAX_SEQ_LENGTH, subset=split, full_edge_features=args.edge_features == 'full',
                        random_demands=demand_mode == 'random'
                    ))

        # test_sets = []
        # for pattern in config['eval_pattern']:
        #     for demand_mode in config['eval_demand_mode']:
        #         test_sets.append(ChlorineDataset(
        #             wdn=config['eval_wdn'], pattern=pattern, path_to_data=DATA_DIR,
        #             timesteps=MAX_SEQ_LENGTH, subset='test', full_edge_features=args.edge_features == 'full',
        #             random_demands=demand_mode == 'random'
        #         ))

        train_split, train_attrib_maxima = concat_datasets(train_valid_sets['train'])
        valid_split, _ = concat_datasets(train_valid_sets['valid'], attrib_maxima=train_attrib_maxima)
        # test_split = ConcatDataset(test_sets)

        train_loader = DataLoader(train_split, shuffle=True, batch_size=config['batch_size'])
        valid_loader = DataLoader(valid_split, shuffle=False, batch_size=config['batch_size'])
        # test_loader = DataLoader(test_split, shuffle=False, batch_size=config['batch_size'])
        return train_loader, valid_loader, train_attrib_maxima#, test_loader

    # torch_ds = ChlorineDataset(wdn=config['wdn'], timesteps=MAX_SEQ_LENGTH, subset='train', full_edge_features=False)
    # test_ds = ChlorineDataset(wdn=config['wdn'], timesteps=MAX_SEQ_LENGTH, subset='test', full_edge_features=False)
    # data_loader = DataLoader(torch_ds, shuffle=True, batch_size=config['batch_size'])
    # test_loader = DataLoader(test_ds, shuffle=False, batch_size=config['batch_size'])

    train_loader, valid_loader, train_attrib_maxima = make_datasplits(config)

    ode_function = PDEFunction(config['hidden_dim'], config['hidden_dim'], config, config['device'])
    model = PDEModel(ode_function, config).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = torch.nn.L1Loss()

    model.set_eval_times(T)

    # ================= Train the model =================
    train_losses, valid_losses = train(
        train_loader, valid_loader, model, epochs=config['epochs'], 
        optimizer=optimizer, loss_fn=loss_fn, dt=config['step_size'], T=config['time']
    )

    eval_metric_fns = [loss_fn]

    # Dump model checkpoint, some results metrics and visualizations.
    results_dir = f'{config["train_wdn"]}_{config["eval_wdn"]}_{config["train_timesteps"]}_'\
                  f'{args.edge_features}_'\
                  f'{",".join(config["train_pattern"])}_'\
                  f'{",".join(config["train_demand_mode"])}'
    
    results_dir = os.path.join(RESULTS_DIR, results_dir)
    os.makedirs(results_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(results_dir, 'model_state.pt'))
    np.save(os.path.join(results_dir, 'train_attrib_maxima.npy'), train_attrib_maxima)
    
    # ================= Evaluate the model =================
    eval_results = {}

    for demand_mode in ['normal', 'random']:
        eval_results[demand_mode] = {}
        for pattern in ['spike', 'wave', 'random']:
            eval_results[demand_mode][pattern] = {}
            # metrics_file.write(f'Demand Mode: "{demand_mode}" - Cl Patterns: "{pattern}"\n')
            test_set = ChlorineDataset(
                wdn=config['eval_wdn'], pattern=pattern, path_to_data=DATA_DIR,
                timesteps=MAX_SEQ_LENGTH, subset='test', full_edge_features=args.edge_features == 'full',
                random_demands=demand_mode == 'random'
            )
            test_set, _ = concat_datasets([test_set], attrib_maxima=train_attrib_maxima)
            test_loader = DataLoader(test_set, shuffle=False, batch_size=config['batch_size'])

            eval_metrics, test_loss = evaluate(
                test_loader, model, metrics=eval_metric_fns, 
                dt=config['step_size'], T=config['evaluation_time']
            )

            for metric, value in eval_metrics.items():#, eval_stds):
                #metric_name = m.__name__ if hasattr(m, '__name__') else m.__class__.__name__
                eval_results[demand_mode][pattern][metric] = value #(f'{metric}\t{value} (± --)\n')
    
    f_out = os.path.join(results_dir, 'eval_metrics.bin')
    dump(eval_results, f_out, compress=True)

    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        json.dump(config, f)

    plt.plot(np.linspace(0, config['epochs'], len(train_losses)), train_losses, label='Training')
    plt.plot(np.linspace(0, config['epochs'], len(valid_losses)), valid_losses, label='Validation')
    plt.legend()
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss (MAE)')
    plt.gca().spines['top'].set_visible(False);plt.gca().spines['right'].set_visible(False)
    plt.grid()
    plt.savefig(os.path.join(results_dir, 'optimization_convergence.pdf'))

    data = next(iter(test_loader))
    model.set_eval_times(config['evaluation_time']-1)
    boundary_condition = partial(
        reservoir_boundary_condition,
        boundary_values=data.x[...,:config['evaluation_time']],
        boundary_index=data.boundary_index[...,:config['evaluation_time']],
        edge_values=data.edge_features[...,:config['evaluation_time']],
        dt=config['step_size'], T=config['evaluation_time'], f=model.m1
    )

    with torch.no_grad():
        pred = model(data.x[...,:1], data.edge_index, edge_weight=None, boundary_condition=boundary_condition, boundary_index=data.boundary_index)
    
    test_ds = test_loader.dataset.datasets[0]
    pred = pred[data.eval_index]
    pred = pred.cpu().numpy() * test_ds.quality_max
    ground_truth = data.y[:,:config['evaluation_time']].cpu().detach().numpy() * test_ds.quality_max

    nt = test_ds.topologies[0]
    G = nx.Graph(nt)
    node_map = dict(zip(G.nodes, range(len(G))))
    node_map_inv = { v : k for k, v in node_map.items() }
    pos = { node_map[n] : nt.get_node_info(n)['coord'] for n in nt }
    G = nx.relabel_nodes(G, node_map)

    test_examples_out = os.path.join(results_dir, 'TestTimesteps')
    test_network_plots_out = os.path.join(test_examples_out, 'Network')
    os.makedirs(test_network_plots_out, exist_ok=True)
    plot_timestep = 30

    for i in range(config['batch_size']):
        fig, ax = plt.subplots(1, 2, figsize=(18, 4))
        nc = pred[i*test_ds.num_nodes:(i+1)*test_ds.num_nodes, plot_timestep]
        nx.draw(G, pos=pos, node_color=nc, with_labels=True, vmin=0, vmax=1, ax=ax[0])
        nc = ground_truth[i*test_ds.num_nodes:(i+1)*test_ds.num_nodes, plot_timestep]
        nx.draw(G, pos=pos, node_color=nc, with_labels=True, vmin=0, vmax=1, ax=ax[1])
        fig.savefig(os.path.join(
            test_network_plots_out, 
            f'{config["eval_wdn"]}_sample_{i}_timestep_{plot_timestep}.pdf'
        ))
        plt.close()
    
    def smooth(x, factor=10):
        return np.convolve(x, np.ones(factor) / factor, mode='same')

    test_timeseries_plots_out = os.path.join(test_examples_out, 'Timeseries')
    os.makedirs(test_timeseries_plots_out, exist_ok=True)

    for i in range(test_ds.num_nodes):
        fig, ax = plt.subplots(figsize=(18, 4))
        ax.set_title(f'Node {node_map_inv[i]}')
        _prediction = pred[i,:]
        _ground_truth = ground_truth[i,:]
        xs = np.arange(0, len(_prediction) * 0.5, 0.5)
        ax.plot(xs, smooth(_ground_truth), label='Ground Truth')
        ax.plot(xs, smooth(_prediction), label='Prediction')
        plt.legend()
        ax.set_xlabel('Time [Hours]') 
        ax.set_ylabel('Chlorine Concentration [mg/L]')

        nx_ax = fig.add_axes((0.9, 0.55, 0.08, 0.24))
        nx_ax.set_title('Node Location')
        nx.draw(
            G, pos=pos, node_color=np.eye(test_ds.num_nodes)[i], cmap='gnuplot', vmax=1.1, 
            ax=nx_ax, node_size=30, alpha=0.6, edge_color='gray', width=3.5
        )
        fig.savefig(os.path.join(
            test_timeseries_plots_out, 
            f'{config["eval_wdn"]}_sample_{i}_steps_{config["evaluation_time"]}.pdf'
        ))
        plt.close()