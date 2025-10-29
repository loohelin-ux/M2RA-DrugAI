# M2RA-DrugAI Framework: Quick Start Example
# This script demonstrates the end-to-end workflow of the M2RA-DrugAI package.

from m2ra_drugai.data.builder import RAGData
from m2ra_drugai.models.m2ra_gnn import M2RAGNN
from m2ra_drugai.optimizer.hierarchical_bo import HierarchicalBO
import torch
from torch.optim import Adam

# --------------------------
# 1. 构建反应-活性图（RAG）数据
# --------------------------
# 示例数据路径（用户需替换为实际数据）
molecules_path = "data/sample_molecules.smi"  # SMILES格式分子数据
reactions_path = "data/sample_reactions.csv"  # 反应条件数据
activities_path = "data/sample_activities.csv"  # 生物活性数据

# 初始化RAG数据结构
rag_data = RAGData(
    molecule_file=molecules_path,
    reaction_file=reactions_path,
    activity_file=activities_path
)
# 预处理：构建异构图（节点/边类型定义、特征初始化）
rag_data.preprocess()
print(f"构建完成RAG图 - 节点数: {rag_data.num_nodes()}, 边数: {rag_data.num_edges()}")

# --------------------------
# 2. 初始化M²RA-GNN模型
# --------------------------
model = M2RAGNN(
    node_feature_dim=64,  # 节点特征维度（根据实际数据调整）
    edge_type_num=3,      # 边类型数量（化学转化/条件调控/分子-靶点作用）
    hidden_dim=128,
    eqgat_head_num=4      # EQGAT层注意力头数
)

# --------------------------
# 3. 模型训练（简化示例）
# --------------------------
optimizer = Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()  # 多任务可扩展为联合损失函数

# 模拟训练循环（10个epoch）
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播：预测产率和活性
    outputs = model(rag_data.get_graph_data())  # 获取图数据（PyTorch Geometric格式）
    pred_yield = outputs["yield"]
    pred_activity = outputs["activity"]
    
    # 计算损失（假设rag_data包含标签）
    loss = criterion(pred_yield, rag_data.yield_labels) + \
           criterion(pred_activity, rag_data.activity_labels)
    
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")

# --------------------------
# 4. 分层贝叶斯优化（双环优化）
# --------------------------
# 定义搜索空间（示例：内圈反应条件+外圈分子结构参数）
search_space = {
    # 内圈：反应条件
    "temperature": (20.0, 150.0),  # 温度范围（℃）
    "solvent": ["water", "DMSO", "ethanol"],  # 溶剂选项
    # 外圈：分子结构参数（示例：环数/官能团类型）
    "ring_count": (1, 5),
    "functional_group": ["-OH", "-NH2", "-COOH"]
}

# 初始化优化器
bo_optimizer = HierarchicalBO(
    model=model,
    search_space=search_space,
    initial_data=rag_data  # 初始数据用于构建代理模型
)

# 生成下一批实验建议（5个候选）
suggestions = bo_optimizer.suggest(n_suggestions=5)
print("\n推荐的实验方案:")
for i, sugg in enumerate(suggestions, 1):
    print(f"方案 {i}: {sugg}")
