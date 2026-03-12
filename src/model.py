import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import HeteroConv, SAGEConv, GCN2Conv

class SHA256SolverGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers=6, alpha=0.1, theta=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.alpha = alpha  # 初始残差的保留比例 (GCNII核心)
        self.theta = theta  # 权重衰减比例

        # 1. 特征编码层 (Embedding)
        # 变量特征：[Nonce, Message, State, Degree] -> 4维
        self.v_embed = Linear(4, hidden_channels)
        # 子句特征：[Length] -> 1维
        self.c_embed = Linear(1, hidden_channels)

        # 2. 异构卷积层堆叠
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            # 使用异构卷积处理：正向连接(pos)、负向连接(neg)和双向推导
            conv = HeteroConv({
                ('variable', 'pos_in', 'clause'): SAGEConv(hidden_channels, hidden_channels),
                ('variable', 'neg_in', 'clause'): SAGEConv(hidden_channels, hidden_channels),
                ('clause', 'rev_pos_in', 'variable'): SAGEConv(hidden_channels, hidden_channels),
                ('clause', 'rev_neg_in', 'variable'): SAGEConv(hidden_channels, hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        # 3. 输出头：映射到变量的活跃度分数
        self.final_lin = Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        # A. 获取初始特征 h0 (GCNII 的锚点)
        v_h0 = F.relu(self.v_embed(x_dict['variable']))
        c_h0 = F.relu(self.c_embed(x_dict['clause']))
        
        v_h, c_h = v_h0, c_h0

        # B. 深度消息传递循环
        for i in range(self.num_layers):
            # 记录上一层状态
            v_old, c_old = v_h, c_h
            
            # 异构卷积
            out_dict = self.convs[i]({'variable': v_h, 'clause': c_h}, edge_index_dict)
            
            # C. 融入 GCNII 思想的初始残差连接 (Initial Residual Connection)
            # 核心公式：h_new = (1-alpha)*h_conv + alpha*h_0
            v_h = (1 - self.alpha) * out_dict['variable'] + self.alpha * v_h0
            c_h = (1 - self.alpha) * out_dict['clause'] + self.alpha * c_h0
            
            # 残差连接与激活
            v_h = F.relu(v_h + v_old)
            c_h = F.relu(c_h + c_old)

        # D. 最终 Readout (只对变量节点进行预测)
        return self.final_lin(v_h)