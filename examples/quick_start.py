# 示例流程：从数据构建到模型训练再到优化
from m2ra_drugai.data.builder import RAGData
from m2ra_drugai.models.m2ra_gnn import M2RAGNN
from m2ra_drugai.optimizer.hierarchical_bo import HierarchicalBO

# 1. 构建RAG图（需补充数据路径/格式说明）
rag_data = RAGData.from_files(molecules="data/mols.sdf", reactions="data/reactions.csv", activities="data/activities.csv")

# 2. 初始化模型
model = M2RAGNN(input_dim=128, hidden_dim=256, num_relation_types=5)

# 3. 训练模型（需补充损失函数、优化器等）
model.train(rag_data, epochs=50, lr=0.001)

# 4. 启动分层贝叶斯优化
bo_optimizer = HierarchicalBO(model, search_space={"temperature": (20, 100), "solvent": ["water", "DMSO"]})
suggestions = bo_optimizer.suggest(n_suggestions=5)  # 建议下一批实验
