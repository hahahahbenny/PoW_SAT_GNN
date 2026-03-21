import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import HeteroConv, SAGEConv

class HGCNIISolver(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers=8, alpha=0.1, theta=0.5, use_inter_layer_res=True):
        super().__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        self.theta = theta
        self.use_inter_layer_res = use_inter_layer_res

        self.v_embed = Linear(4, hidden_channels)
        self.c_embed = Linear(1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.v_weights = torch.nn.ModuleList([Linear(hidden_channels, hidden_channels) for _ in range(num_layers)])

        for _ in range(num_layers):
            self.convs.append(HeteroConv({
                ('variable', 'pos_in', 'clause'): SAGEConv(hidden_channels, hidden_channels),
                ('variable', 'neg_in', 'clause'): SAGEConv(hidden_channels, hidden_channels),
                ('clause', 'rev_pos_in', 'variable'): SAGEConv(hidden_channels, hidden_channels),
                ('clause', 'rev_neg_in', 'variable'): SAGEConv(hidden_channels, hidden_channels),
            }, aggr='sum'))

        self.final_lin = Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        v_h0 = F.relu(self.v_embed(x_dict['variable']))
        c_h0 = F.relu(self.c_embed(x_dict['clause']))
        v_h, c_h = v_h0, c_h0

        for i in range(self.num_layers):
            v_old = v_h
            # 1. 异构卷积
            out = self.convs[i]({'variable': v_h, 'clause': c_h}, edge_index_dict)
            v_h, c_h = out['variable'], out['clause']

            # 2. 初始残差 (GCNII核心)
            v_h = (1 - self.alpha) * v_h + self.alpha * v_h0
            c_h = (1 - self.alpha) * c_h + self.alpha * c_h0

            # 3. 恒等映射 (Identity Mapping)
            # 优化点：预指定 device 避免同步
            beta = torch.log(torch.tensor(self.theta / (i + 1) + 1, device=v_h.device))
            v_h = (1 - beta) * v_h + beta * self.v_weights[i](v_h)

            # 4. 层间残差 (消融点)
            if self.use_inter_layer_res:
                v_h = v_h + v_old
            
            v_h, c_h = F.relu(v_h), F.relu(c_h)

        return self.final_lin(v_h)