from torch_geometric.nn import GCNConv
from torch import nn
import torch


class PDEFunction(GCNConv):

    def __init__(self, in_features, out_features, config, device):
        super().__init__(1, 1, bias=False, aggr='sum')
        self.linear1 = torch.nn.Linear(config['n_edge_features'] + config['hidden_dim'], config['hidden_dim'], bias=False)
        self.linear2 = torch.nn.Linear(config['hidden_dim'], config['hidden_dim'], bias=False)
        self.linear3 = torch.nn.Linear(config['hidden_dim'], config['hidden_dim'], bias=False)
        self.alpha_train = nn.Parameter(torch.tensor(1.0))
        self.beta_train = nn.Parameter(torch.tensor(0.0))
        self.config = config

    def init(self, x, edge_index, edge_weight=None, boundary_condition=None, boundary_index=None):
        num_nodes = x.shape[0]
        self.boundary_condition = boundary_condition
        self.boundary_index = boundary_index
        ode_edge_index = edge_index
        return x, ode_edge_index, {
            'x0' : x,
        }
    
    def message(self, x_j, x_i, edge_features=None):
        return self.linear2(torch.nn.functional.relu(self.linear1(torch.cat((x_j - x_i, edge_features), dim=-1))))
   
    # the t param is needed by the ODE solver.
    def forward(self, t, x, edge_index, edge_features=None, x0=None):
        x, edge_features = self.boundary_condition(
            t, x, edge_features
        )
        
        ax = self.propagate(
            edge_index, x=x, edge_features=edge_features, size=None
        )

        if self.config.get('alpha_sigmoid'):
            alpha = torch.sigmoid(self.alpha_train)
        else:
            alpha = self.alpha_train

        # x_norm = torch.norm(x, dim=-1, keepdim=True)
        # update = self.linear3(torch.cat((ax, x), dim=-1))
        # xu = x + update
        # xunorm = torch.norm(xu, dim=-1, keepdim=True)
        # norm_pres_update = update - (xu/(xunorm+1e-5) * (xunorm - x_norm))

        f = (alpha * (ax - x) + self.linear3(x)).clone()
        f[self.boundary_index] = 0.
        
        return f