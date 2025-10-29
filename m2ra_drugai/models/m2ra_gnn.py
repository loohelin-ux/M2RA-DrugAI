"""
M²RA-GNN (Multi-Modal Reaction-Activity Graph Neural Network) Model Definition.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch_geometric.data import HeteroData

class EQGATLayer(torch.nn.Module):
    """等变图注意力层（处理分子3D几何信息）"""
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads)
        self.linear = Linear(out_dim, out_dim)

    def forward(self, x, edge_index, pos):
        """
        x: 节点特征
        pos: 节点3D坐标（用于几何信息编码）
        """
        # 融合几何信息（位置差的L2范数）
        row, col = edge_index
        pos_diff = pos[row] - pos[col]
        geo_feat = torch.norm(pos_diff, dim=-1, keepdim=True)  # 边的几何特征
        x = x + 0.1 * geo_feat.mean(dim=0)  # 简化的几何信息融合
        
        # GAT注意力传播
        x = self.gat(x, edge_index)
        return self.linear(x)

class M2RAGNN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_type_num, hidden_dim, eqgat_head_num=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 节点特征投影（统一不同类型节点的特征维度）
        self.node_proj = Linear(node_feature_dim, hidden_dim)
        
        # 异构消息传递层（针对不同边类型）
        self.hetero_conv1 = HeteroConv({
            "chemical_transformation": GATConv(hidden_dim, hidden_dim // 2, heads=2),
            "condition_modulation": GATConv(hidden_dim, hidden_dim // 2, heads=2),
            "molecule_target": EQGATLayer(hidden_dim, hidden_dim, heads=eqgat_head_num)
        }, aggr="sum")
        
        self.hetero_conv2 = HeteroConv({
            "chemical_transformation": GATConv(hidden_dim, hidden_dim // 2, heads=2),
            "condition_modulation": GATConv(hidden_dim, hidden_dim // 2, heads=2),
            "molecule_target": EQGATLayer(hidden_dim, hidden_dim, heads=eqgat_head_num)
        }, aggr="sum")
        
        # 多任务预测头
        self.yield_head = Linear(hidden_dim, 1)  # 产率预测
        self.activity_head = Linear(hidden_dim, 1)  # 活性预测

    def forward(self, graph_data: HeteroData):
        """
        graph_data: 异构图数据（包含节点特征、边索引、3D坐标等）
        """
        # 1. 初始化节点特征
        x_dict = {
            node_type: self.node_proj(graph_data[node_type].x)
            for node_type in graph_data.node_types
        }
        
        # 2. 异构消息传递（带激活函数）
        x_dict = self.hetero_conv1(x_dict, graph_data.edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        
        x_dict = self.hetero_conv2(x_dict, graph_data.edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        
        # 3. 多任务预测（取产物分子节点的特征进行预测）
        product_features = x_dict["product"]  # 假设节点类型包含"product"
        pred_yield = self.yield_head(product_features).squeeze()
        pred_activity = self.activity_head(product_features).squeeze()
        
        return {"yield": pred_yield, "activity": pred_activity}
