"""反应-活性图（RAG）构建器"""
import torch
from torch_geometric.data import HeteroData

class RAGDataBuilder:
    def __init__(self):
        self.node_types = ["molecule", "reaction", "condition", "target", "activity"]
        self.edge_types = [("molecule", "participates_in", "reaction"),
                          ("condition", "regulates", "reaction"),
                          ("reaction", "generates", "molecule"),
                          ("molecule", "binds_to", "target"),
                          ("target", "has", "activity")]

    def build(self, samples):
        """
        输入：samples = [(反应物SMILES, 产物SMILES, 反应条件dict, 靶点序列, 活性值), ...]
        输出：HeteroData格式的RAG图
        """
        rag_data = HeteroData()
        mol_smiles = []
        reaction_info = []
        condition_info = []
        target_seqs = []
        activity_values = []

        # 解析输入样本
        for reactant, product, cond, target, act in samples:
            mol_smiles.extend([reactant, product])
            reaction_info.append(f"{reactant}→{product}")
            condition_info.append(cond)
            target_seqs.append(target)
            activity_values.append(act)

        # 为节点分配特征
        rag_data["molecule"].x = torch.randn(len(mol_smiles), 64)  # 分子特征（64维）
        rag_data["reaction"].x = torch.randn(len(reaction_info), 64)  # 反应特征
        rag_data["condition"].x = torch.randn(len(condition_info), 32)  # 条件特征
        rag_data["target"].x = torch.randn(len(target_seqs), 64)  # 靶点特征
        rag_data["activity"].x = torch.tensor(activity_values).unsqueeze(1).float()  # 活性值

        # 标签（用于训练）
        rag_data.yield_labels = torch.randn(len(reaction_info))  # 模拟产率标签
        rag_data.activity_labels = torch.tensor(activity_values).float()  # 活性标签

        # 标记节点类型（适配模型调用）
        rag_data.mol_node_mask = torch.ones(len(mol_smiles), dtype=torch.bool)
        rag_data.reaction_node_mask = torch.ones(len(reaction_info), dtype=torch.bool)
        rag_data.target_node_mask = torch.ones(len(target_seqs), dtype=torch.bool)

        return rag_data

    def get_search_space(self):
        """返回优化搜索空间（适配BO优化器）"""
        return {
            "molecules": ["CCO", "CCN", "CC(=O)O", "CC(=O)N", "CC(=O)OCC"],
            "conditions": {"temperature": (25, 80), "solvent": ["水", "乙醇", "1,4-二氧六环"]}
        }
