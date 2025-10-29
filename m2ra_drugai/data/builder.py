"""
RAG (Reaction-Activity Graph) Data Structure Builder.
"""

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from rdkit import Chem

class RAGData:
    def __init__(self, molecule_file, reaction_file, activity_file):
        self.molecule_file = molecule_file
        self.reaction_file = reaction_file
        self.activity_file = activity_file
        
        # 存储图数据和标签
        self.graph = HeteroData()
        self.yield_labels = None
        self.activity_labels = None

    def preprocess(self):
        """构建RAG图的核心方法"""
        # 1. 加载分子数据（SMILES格式）
        molecules = pd.read_csv(self.molecule_file, header=None, names=["smiles"])
        self._add_molecule_nodes(molecules)
        
        # 2. 加载反应数据（前体->产物+条件）
        reactions = pd.read_csv(self.reaction_file)
        self._add_reaction_edges(reactions)
        
        # 3. 加载活性数据（分子->靶点+活性值）
        activities = pd.read_csv(self.activity_file)
        self._add_activity_edges(activities)
        
        # 4. 初始化标签（示例：取产物分子的产率和活性）
        self.yield_labels = torch.tensor(reactions["yield"].values, dtype=torch.float)
        self.activity_labels = torch.tensor(activities["activity"].values, dtype=torch.float)

    def _add_molecule_nodes(self, molecules_df):
        """添加分子节点（前体/产物）并提取特征"""
        # 简化：用分子的原子数作为特征（实际应使用Morgan指纹等）
        features = []
        for smi in molecules_df["smiles"]:
            mol = Chem.MolFromSmiles(smi)
            features.append([mol.GetNumAtoms()])  # 示例特征
        
        self.graph["molecule"].x = torch.tensor(features, dtype=torch.float)
        self.graph["product"].x = self.graph["molecule"].x  # 假设部分分子是产物

    def _add_reaction_edges(self, reactions_df):
        """添加反应边（化学转化+条件调控）"""
        # 化学转化边：前体 -> 产物
        precursor_indices = reactions_df["precursor_id"].values
        product_indices = reactions_df["product_id"].values
        self.graph["molecule", "chemical_transformation", "product"].edge_index = torch.tensor([
            precursor_indices, product_indices
        ], dtype=torch.long)
        
        # 条件调控边：反应条件 -> 化学转化（简化示例）
        self.graph["condition", "condition_modulation", "chemical_transformation"].edge_index = torch.tensor([
            reactions_df["condition_id"].values,
            np.arange(len(reactions_df))  # 假设每条边对应一个反应
        ], dtype=torch.long)

    def _add_activity_edges(self, activities_df):
        """添加分子-靶点作用边"""
        self.graph["product", "molecule_target", "target"].edge_index = torch.tensor([
            activities_df["product_id"].values,
            activities_df["target_id"].values
        ], dtype=torch.long)

    def get_graph_data(self):
        """返回PyTorch Geometric格式的异构图数据"""
        return self.graph

    def get_search_space_samples(self):
        """为BO提供初始搜索空间样本（简化示例）"""
        # 实际应从反应条件和分子参数中提取
        return np.random.rand(10, len(self.search_space))  # 10个初始样本

    def num_nodes(self):
        return sum(self.graph.num_nodes_dict.values())

    def num_edges(self):
        return sum(self.graph.num_edges_dict.values())
