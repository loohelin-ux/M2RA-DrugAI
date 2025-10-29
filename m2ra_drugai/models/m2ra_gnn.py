"""多模态反应-活性图神经网络（M²RA-GNN）"""
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class EQGATLayer(pyg_nn.MessagePassing):
    """3D几何感知等变图注意力层"""
    def __init__(self, hidden_dim):
        super().__init__(aggr="add")
        self.att = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.update = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, pos, edge_index):
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_i, x_j, pos_i, pos_j):
        dist = pos_i - pos_j
        att_score = self.att(torch.cat([x_i, x_j, dist], dim=1))
        att_score = torch.softmax(att_score, dim=0)
        return x_j * att_score

    def update(self, aggr_out):
        return self.update(aggr_out)

class M2RAGNN(nn.Module):
    def __init__(self, node_feat_dim=64, edge_feat_dim=32, hidden_dim=128, num_relation_types=3):
        super().__init__()
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)
        self.relational_conv = nn.ModuleList([
            pyg_nn.GATConv(hidden_dim, hidden_dim) for _ in range(num_relation_types)
        ])
        self.eqgat = EQGATLayer(hidden_dim)
        self.yield_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.activity_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, rag_data):
        # 分子3D特征处理
        mol_feats = self.node_encoder(rag_data["molecule"].x)
        pos = torch.randn_like(mol_feats[:, :3])  # 模拟3D坐标
        mol_feats = self.eqgat(mol_feats, pos, torch.tensor([[0,1],[1,0]]).long())

        # 异构消息传递
        reaction_feats = self.node_encoder(rag_data["reaction"].x)
        target_feats = self.node_encoder(rag_data["target"].x)

        # 融合特征并预测
        pred_yield = self.yield_head(reaction_feats).squeeze()
        pred_activity = self.activity_head(target_feats).squeeze()
        return {"yield": pred_yield, "activity": pred_activity}
