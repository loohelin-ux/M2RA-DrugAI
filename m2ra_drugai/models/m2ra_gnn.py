"""
M²RA-GNN: 实现THEORY.md中定义的边类型专属消息传递
φ_化学转化, φ_条件调控, φ_分子-靶点作用
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch_geometric.nn.norm import BatchNorm

class EQGATLayer(torch.nn.Module):
    """等变图注意力层（处理分子3D几何信息）"""
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads)
        self.proj = Linear(out_dim, out_dim)

    def forward(self, x, edge_index, pos):
        # 计算边的几何特征（位置差的L2范数）
        row, col = edge_index
        pos_diff = pos[row] - pos[col]
        geo_feat = torch.norm(pos_diff, dim=-1).unsqueeze(1)  # [E, 1]
        # 融合几何信息到节点特征
        x = x + 0.1 * geo_feat.mean(dim=0)  # 全局几何偏置
        # GAT注意力传播
        x = self.gat(x, edge_index)
        return self.proj(x)

class M2RAGNN(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # 输入特征投影（适配不同节点类型的特征维度）
        self.precursor_proj = Linear(2049, hidden_dim)  # 2048摩根指纹+1个坐标特征
        self.product_proj = Linear(2049, hidden_dim)
        self.condition_proj = Linear(5, hidden_dim)     # 1温度+4溶剂独热（示例）
        self.target_proj = Linear(1, hidden_dim)

        # 异构消息传递层（边类型专属φ函数）
        self.conv1 = HeteroConv({
            ("precursor", "chemical_transformation", "product"): GATConv(hidden_dim, hidden_dim, heads=2),
            ("condition", "condition_modulation", "chemical_transformation"): GATConv(hidden_dim, hidden_dim, heads=2),
            ("product", "molecule_target", "target"): EQGATLayer(hidden_dim, hidden_dim, heads=2)
        }, aggr="sum")

        self.conv2 = HeteroConv({
            ("precursor", "chemical_transformation", "product"): GATConv(hidden_dim, hidden_dim, heads=2),
            ("condition", "condition_modulation", "chemical_transformation"): GATConv(hidden_dim, hidden_dim, heads=2),
            ("product", "molecule_target", "target"): EQGATLayer(hidden_dim, hidden_dim, heads=2)
        }, aggr="sum")

        # 批归一化
        self.bn = BatchNorm(hidden_dim)

        # 预测头
        self.yield_head = Linear(hidden_dim, 1)
        self.activity_head = Linear(hidden_dim, 1)

    def forward(self, graph):
        # 1. 节点特征投影
        x_dict = {
            "precursor": self.precursor_proj(graph["precursor"].x),
            "product": self.product_proj(graph["product"].x),
            "condition": self.condition_proj(graph["condition"].x),
            "target": self.target_proj(graph["target"].x)
        }

        # 2. 异构消息传递（带激活）
        x_dict = self.conv1(x_dict, graph.edge_index_dict)
        x_dict = {k: self.bn(F.relu(v)) for k, v in x_dict.items()}
        
        x_dict = self.conv2(x_dict, graph.edge_index_dict)
        x_dict = {k: self.bn(F.relu(v)) for k, v in x_dict.items()}

        # 3. 多任务预测
        # 产率预测（基于化学转化边的聚合特征）
        reaction_feat = x_dict[("precursor", "chemical_transformation", "product")]
        pred_yield = self.yield_head(reaction_feat).squeeze()
        
        # 活性预测（基于分子-靶点作用边的特征）
        activity_feat = x_dict[("product", "molecule_target", "target")]
        pred_activity = self.activity_head(activity_feat).squeeze()

        return {"yield": pred_yield, "activity": pred_activity}
