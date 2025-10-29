(https://github.com/loohelin-ux/M2RA-DrugAI/blob/main/LICENSE) (https://www.python.org/) ((https://www.google.com/search?q=https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg))](https://pytorch.org/) ((https://www.google.com/search?q=https://img.shields.io/badge/Status-Prototyping-green.svg))]((https://www.google.com/url?sa=E&source=gmail&q=https://github.com/loohelin-ux/M2RA-DrugAI))
M2RA-DrugAI: A PyTorch Framework for Multi-Modal Reaction-Activity Drug Discovery
M2RA-DrugAI is an in-development framework for the project "Multi-Modal Scientific Data Unified Graph Representation Theory and Multi-Objective Active Learning Algorithm Research" (Hubei Provincial Natural Science Foundation, PI: Helin Lu).

The core of this framework is the implementation of the Reaction-Activity Graph (RAG) theory, providing an end-to-end solution to bridge the gap between molecular design and synthetic accessibility in drug discovery.

Core Concepts
Unified Data Representation: Utilizes a RAGData class to represent heterogeneous information (molecules, reaction conditions, biological activity) in a single graph structure. See the(https://www.google.com/search?q=m2ra_drugai/data/DATA_SCHEMA.md) for details.

Advanced GNN Model: Implements the M²RA-GNN, a novel heterogeneous graph neural network designed to learn from the RAG structure. See the Model Plan.

Dual-Loop Optimization: Integrates a Hierarchical Bayesian Optimization module for intelligent, multi-objective active learning. See the Optimizer Plan.

Project Status
The theoretical foundation has been laid out, and the software architecture is currently being prototyped.

(https://www.google.com/search?q=THEORY.md)

(https://www.google.com/search?q=experiments/run_01_config.yml)

(https://www.google.com/search?q=notebooks/01_Initial_Data_Exploration.ipynb)

How to Cite
If your research work uses concepts from this project, please cite:

@misc{Lu_M2RA-DrugAI_2025,
  author = {Lu, Helin and Xu, Yongrui and Team},
  title = {{M2RA-DrugAI: A PyTorch Framework for Multi-Modal Reaction-Activity Drug Discovery}},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/loohelin-ux/M2RA-DrugAI}}
}
