# M2RA-DrugAI


```
\# M2RA-DrugAI

\[!\[PyTorch 2.0+]\(https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)]\(https://pytorch.org/get-started/locally/)

\[!\[PyTorch Geometric]\(https://img.shields.io/badge/PyTorch%20Geometric-2.3.0+-7932a8.svg)]\(https://pytorch-geometric.readthedocs.io/)

M2RA-DrugAI 是一个基于 PyTorch 和 PyTorch Geometric 构建的开源框架，专为多模态反应-活性药物发现设计。该框架旨在实现“多模态科学数据的统一图表示理论与多目标主动学习算法研究”项目（湖北省自然科学基金，项目负责人：陆何林）的核心算法，通过创新的\*\*反应-活性图（Reaction-Activity Graph, RAG）\*\* 理论，解决药物发现中“分子设计”与“合成可及性”脱节的关键问题，提供端到端的解决方案。

\## 核心特性

\- \*\*统一数据表示\*\*：通过 \`RAGData\` 类将分子结构、反应条件、生物活性等异构数据统一建模为反应-活性图（RAG），实现多模态信息的有机融合。

&#x20;&#x20;

\- \*\*先进图神经网络\*\*：内置 M²RA-GNN 模型，这是一种专为 RAG 设计的异构图神经网络，支持边类型专属的消息传递机制，能同时学习化学转化规律与分子-靶点作用关系。

\- \*\*双重闭环优化\*\*：集成基于分层贝叶斯优化的主动学习模块，通过“外循环（分子发现）+ 内循环（合成优化）”的双重闭环，协同优化分子的生物活性与合成产率。

\## 理论基础

本框架的核心理论基础是\*\*反应-活性图（RAG）\*\*，其形式化定义、层次化结构及消息传递机制详见 \[THEORY.md]\(THEORY.md)。RAG 创新性地将化学反应实体与功能测试实体纳入统一图结构，为多模态科学数据的联合建模提供了数学基础。

\## 仓库结构
```

M2RA-DrugAI/

├── m2ra\_drugai/                # 核心代码目录

│   ├── data/

│   │   └── builder.py          # RAG 图构建逻辑，实现异构数据到图结构的转换

│   ├── models/

│   │   └── m2ra\_gnn.py         # M²RA-GNN 模型定义，包含异构消息传递与多任务预测

│   └── optimizer/

│       └── hierarchical\_bo.py  # 分层贝叶斯优化器，实现双重闭环主动学习

├── examples/

│   └── quick\_start.py          # 快速入门示例，展示框架端到端工作流

├── THEORY.md                   # 理论文档，详解 RAG 理论与 M²RA-GNN 原理

├── LICENSE                     # Apache 2.0 许可证

└── README.md                   # 项目说明文档



```
\## 安装指南

\### 环境依赖

\- Python 3.8+

\- PyTorch 2.0+

\- PyTorch Geometric 2.3.0+

\- RDKit 2023.03+（分子解析工具）

\- Scipy 1.10+（优化算法支持）

\### 安装步骤

1\. 克隆仓库：

&#x20;  \`\`\`bash

&#x20;  git clone https://github.com/loohelin-ux/M2RA-DrugAI.git

&#x20;  cd M2RA-DrugAI
```



1. 创建并激活虚拟环境（推荐）：



```
conda create -n m2ra-env python=3.9

conda activate m2ra-env
```



1. 安装依赖包：



```
\# 安装PyTorch（根据CUDA版本调整，此处为CPU版示例）

pip install torch==2.0.1

\# 安装PyTorch Geometric及相关依赖

pip install torch\_geometric==2.3.0

pip install pyg\_lib torch\_scatter torch\_sparse torch\_cluster torch\_spline\_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

\# 安装其他依赖

pip install rdkit-pypi==2023.3.3 scipy==1.10.1
```



1. 安装本框架（开发模式）：



```
pip install -e .
```



1. 验证安装：



```
python examples/quick\_start.py

\# 若输出框架工作流程日志，则安装成功
```

## 快速开始

运行 `examples/``quick_start.py` 可快速体验框架核心功能，包括：



* RAG 图的构建过程

* M²RA-GNN 模型的训练流程

* 分层贝叶斯优化器的实验推荐逻辑

示例代码展示了从异构数据到图结构的转换、模型训练及优化推荐的完整流程，可作为实际应用的基础模板。

## 引用方式

如果您的研究使用了本框架，请引用以下内容：



```
@misc{Lu\_M2RA-DrugAI\_2025,

&#x20; author = {Lu, Helin and Xu, Yongrui and Team},

&#x20; title = {{M2RA-DrugAI: A PyTorch Framework for Multi-Modal Reaction-Activity Drug Discovery}},

&#x20; year = {2025},

&#x20; publisher = {GitHub},

&#x20; journal = {GitHub repository},

&#x20; howpublished = {\url{https://github.com/loohelin-ux/M2RA-DrugAI}}

}
```

## 许可证

本项目基于 Apache License 2.0 许可证开源，详情参见 [LICENSE](LICENSE)。



```
关于下载方式：您可以直接复制上述代码块中的全部内容，然后在本地创建一个名为 \`README.md\` 的文件，将内容粘贴进去即可完成保存。如果需要通过代码仓库下载，也可以使用前文提到的 \`git clone\` 命令克隆整个仓库获取完整文件。
```
