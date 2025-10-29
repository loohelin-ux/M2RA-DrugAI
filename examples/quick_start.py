"""M2RA-DrugAI 快速开始示例：从数据到优化的完整流程"""

from m2ra_drugai.data.builder import RAGData
from m2ra_drugai.models.m2ra_gnn import M2RAGNN
from m2ra_drugai.optimizer.hierarchical_bo import HierarchicalBO
import torch
import torch.optim as optim

# 1. 构建RAG图数据
rag_data = RAGData(
    molecule_file="data/molecules.csv",
    reaction_file="data/reactions.csv",
    activity_file="data/activities.csv"
)
rag_data.preprocess()
graph = rag_data.get_graph()
print(f"RAG图构建完成：{graph}")

# 2. 初始化模型并训练
model = M2RAGNN(hidden_dim=128)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环（简化）
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    outputs = model(graph)
    loss = criterion(outputs["yield"], graph["chemical_transformation"].yield_label) + \
           criterion(outputs["activity"], graph["molecule_target"].activity_label)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 3. 双环优化
bo_optimizer = HierarchicalBO(model, rag_data)

# 外圈：推荐新分子结构
new_molecules = bo_optimizer.suggest_outer(n=2)
print("推荐的新分子结构参数：", new_molecules)

# 内圈：为第一个分子推荐反应条件
if new_molecules:
    best_conditions = bo_optimizer.suggest_inner(new_molecules[0], n=3)
    print("推荐的反应条件：", best_conditions)
