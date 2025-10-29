"""
RAG (Reaction-Activity Graph) Data Structure Builder.
Implements formal definition from THEORY.md: G = (V, E, τ, ρ)
"""

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from rdkit import Chem
from rdkit.Chem import AllChem

class RAGData:
    def __init__(self, molecule_file, reaction_file, activity_file):
        self.molecule_file = molecule_file
        self.reaction_file = reaction_file
        self.activity_file = activity_file
        self.graph = HeteroData()  # 异构图容器
        self.node_types = ["precursor", "product", "condition", "target"]  # τ映射的节点类型
        self.edge_types = [
            ("precursor", "chemical_transformation", "product"),
            ("condition", "condition_modulation", "chemical_transformation"),
            ("product", "molecule_target", "target")
        ]  # ρ映射的边类型

    def preprocess(self):
        """构建RAG图的完整流程"""
        self._load_molecules()      # 加载分子节点（前体/产物）
        self._load_conditions()     # 加载反应条件节点
        self._load_targets()        # 加载靶点节点
        self._build_reaction_edges()# 构建反应相关边
        self._build_activity_edges()# 构建活性相关边
        self._init_labels()         # 初始化标签

    def _load_molecules(self):
        """加载分子数据并提取特征（摩根指纹+3D坐标）"""
        molecules = pd.read_csv(self.molecule_file)
        # 前体分子
        precursor_smiles = molecules[molecules["type"] == "precursor"]["smiles"].values
        self.graph["precursor"].x = self._get_molecule_features(precursor_smiles)
        # 产物分子
        product_smiles = molecules[molecules["type"] == "product"]["smiles"].values
        self.graph["product"].x = self._get_molecule_features(product_smiles)

    def _get_molecule_features(self, smiles_list):
        """提取分子特征：摩根指纹+原子坐标统计"""
        features = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            # 摩根指纹（2048维）
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_arr = torch.tensor(list(fp), dtype=torch.float)
            # 3D坐标统计（简化：用原子数代替）
            coord_feat = torch.tensor([mol.GetNumAtoms()], dtype=torch.float)
            features.append(torch.cat([fp_arr, coord_feat]))
        return torch.stack(features)

    def _load_conditions(self):
        """加载反应条件（温度、溶剂等）"""
        reactions = pd.read_csv(self.reaction_file)
        conditions = reactions[["temperature", "solvent", "catalyst"]].drop_duplicates()
        # 编码条件特征（温度归一化+溶剂独热编码）
        temp_norm = (conditions["temperature"] - 20) / (150 - 20)  # 归一化到[0,1]
        solvent_onehot = pd.get_dummies(conditions["solvent"]).values
        self.graph["condition"].x = torch.tensor(
            [torch.cat([torch.tensor([t]), torch.tensor(s)]) 
             for t, s in zip(temp_norm, solvent_onehot)],
            dtype=torch.float
        )

    def _load_targets(self):
        """加载靶点节点（简化：用靶点ID作为特征）"""
        activities = pd.read_csv(self.activity_file)
        target_ids = activities["target_id"].unique()
        self.graph["target"].x = torch.tensor([[id] for id in target_ids], dtype=torch.float)

    def _build_reaction_edges(self):
        """构建化学转化边和条件调控边"""
        reactions = pd.read_csv(self.reaction_file)
        # 化学转化边：precursor -> product
        self.graph["precursor", "chemical_transformation", "product"].edge_index = torch.tensor([
            reactions["precursor_id"].values,
            reactions["product_id"].values
        ], dtype=torch.long)
        # 条件调控边：condition -> chemical_transformation
        self.graph["condition", "condition_modulation", "chemical_transformation"].edge_index = torch.tensor([
            reactions["condition_id"].values,
            reactions.index.values  # 用反应索引作为边ID
        ], dtype=torch.long)

    def _build_activity_edges(self):
        """构建分子-靶点作用边"""
        activities = pd.read_csv(self.activity_file)
        self.graph["product", "molecule_target", "target"].edge_index = torch.tensor([
            activities["product_id"].values,
            activities["target_id"].values
        ], dtype=torch.long)

    def _init_labels(self):
        """初始化产率和活性标签"""
        reactions = pd.read_csv(self.reaction_file)
        self.graph["chemical_transformation"].yield_label = torch.tensor(
            reactions["yield"].values, dtype=torch.float
        )
        activities = pd.read_csv(self.activity_file)
        self.graph["molecule_target"].activity_label = torch.tensor(
            activities["activity_score"].values, dtype=torch.float
        )

    def get_graph(self):
        return self.graph
