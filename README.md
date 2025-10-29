(https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/) (https://img.shields.io/badge/Status-Prototyping-green.svg)](https://github.com/loohelin-ux/M2RA-DrugAI)
# M2RA-DrugAI
M2RA-DrugAI: A PyTorch Framework for Multi-Modal Reaction-Activity Drug Discovery
(https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)

M2RA-DrugAI 是一个基于PyTorch和PyTorch Geometric构建的开源软件包，旨在实现“多模态科学数据的统一图表示理论与多目标主动学习算法研究”项目（湖北省自然科学基金，项目负责人：陆何林）的核心算法。本框架的核心是**反应-活性图（Reaction-Activity Graph, RAG）**理论的实现，它为解决药物发现中“分子设计”与“合成可及性”脱节的根本问题提供了一个端到端的解决方案。

核心特性
统一数据表示: 提供了RAGData类，可将分子、反应条件、生物活性等异构信息统一表示为RAG图。

先进的GNN模型: 内置了M²RA-GNN模型，这是一个先进的异构图神经网络，能够直接在RAG上进行学习。

双重闭环优化: 集成了基于分层贝叶斯优化的多目标主动学习模块，能够智能地协同优化分子的合成产率与生物活性。

理论基础
本框架的详细理论基础，包括“反应-活性图”（RAG）的形式化定义，请参阅 THEORY.md 文件。

仓库结构
M2RA-DrugAI/ ├── m2ra_drugai/ │ ├── data/ │ │ └── builder.py # RAG图构建核心逻辑 │ ├── models/ │ │ └── m2ra_gnn.py # M²RA-GNN模型定义 │ └── optimizer/ │ └── hierarchical_bo.py # 双重闭环优化器 ├── examples/ │ └── quick_start.py ├── THEORY.md └── README.md


## 如何引用
如果您的研究工作使用了本软件包，请引用我们的项目：
```bibtex
@misc{Lu_M2RA-DrugAI_2025,
  author = {Lu, Helin and Xu, Yongrui and Team},
  title = {{M2RA-DrugAI: A PyTorch Framework for Multi-Modal Reaction-Activity Drug Discovery}},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/your-username/M2RA-DrugAI](https://github.com/your-username/M2RA-DrugAI)}}
}
