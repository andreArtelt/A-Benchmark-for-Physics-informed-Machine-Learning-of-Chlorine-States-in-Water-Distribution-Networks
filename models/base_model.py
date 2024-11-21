from torchdiffeq import odeint, odeint_adjoint
from functools import partial
import torch.nn.functional as F
import torch

class PDEModel(torch.nn.Module):
    '''
    This class implements everything thats required to integrate an ODE.
    These are mainly the settings for the odeint integrator.
    '''

    def __init__(self, ode_function, config):
        super().__init__()
        self.ode_function = ode_function
        self.method = config['method']
        self.step_size = config['step_size']
        self.atol = config.get('atol', 1e-9)
        self.rtol = config.get('rtol', 1e-7)
        self.adjoint = config.get('adjoint')
        self.adjoint_method = config.get('adjoint_method')
        self.integrator = odeint_adjoint if self.adjoint else odeint
        self.device = config['device']
        self.config = config
        self.set_eval_times(config['time'])
        self.grand_setup_ode(config)
        # Input transformation
        self.m1 = torch.nn.Linear(config['input_dim'], config['hidden_dim'], bias=False)
        # Output transformation
        self.m2 = torch.nn.Linear(config['hidden_dim'], config['output_dim'], bias=False)
        
    @property
    def ode_opt(self):
        opt = {
            'method' : self.method,
            'options' : { 'step_size': self.step_size },
            'atol' : self.atol,
            'rtol' : self.rtol
        }
        if self.adjoint:
            opt.update({
                'adjoint_method' : self.config['adjoint_method'],
                'adjoint_options' : { 'step_size': self.config['adjoint_step_size'] },
                'adjoint_atol' : self.atol_adjoint,
                'adjoint_rtol' : self.rtol_adjoint
            })
        return opt
    
    def grand_setup_ode(self, config):
        self.atol = self.config['tol_scale'] * 1e-7
        self.rtol = self.config['tol_scale'] * 1e-9
        if self.config.get('adjoint'):
            self.atol_adjoint = self.config['tol_scale_adjoint'] * 1e-7
            self.rtol_adjoint = self.config['tol_scale_adjoint'] * 1e-9

    def integrate(self, x, edge_index, **kwargs):
        x0, edge_index, int_kwargs = self.ode_function.init(x, edge_index, **kwargs)
        
        self.ode_function.forward = partial(
            self.ode_function.forward, edge_index=edge_index, **int_kwargs
        )
        
        state_dt = self.integrator(
            self.ode_function, x0, self.eval_times,
            **self.ode_opt
        )
        return state_dt
    
    def forward(self, x, edge_index, edge_weight=None, batch_index=None, boundary_condition=None, boundary_index=None):
        # Only use the first time step as the initial value
        # (Make sure everything except sensor nodes are masked).
        x = x[:,:1]
        x = self.m1(x)
        state_dt = self.integrate(
            x, edge_index, edge_weight=edge_weight, boundary_condition=boundary_condition, boundary_index=boundary_index
        )
        
        z = state_dt.transpose(0, 1)
            
        z = F.relu(z)
        
        if self.config.get('fc_out'):
            z = self.fc(z)
            z = F.relu(z)
            
        z = F.dropout(z, self.config['dropout'], training=self.training)
            
        # Decode each node embedding to get node label.
        z = self.m2(z)
        return z.squeeze(-1)

    def set_eval_times(self, time):
        time = torch.tensor(time + 1, dtype=torch.float32).to(self.device)
        self.eval_times = torch.arange(0, time, dtype=torch.float32).to(self.device)