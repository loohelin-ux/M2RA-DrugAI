理论框架：反应-活性图（RAG）
摘要
在药物发现、材料科学等领域，科学数据天然呈现多模态、跨领域、长因果链的特点。现有的人工智能模型通常将“合成过程”与“功能属性”割裂建模，导致了“设计”与“实现”的脱节。为解决这一根本性瓶颈，我们提出了一种新颖的、统一的图表示理论——反应-活性图（Reaction-Activity Graph, RAG）。RAG是一个层次化的异构图框架，它首次将化学反应中的实体（分子、条件）与功能测试中的实体（生物靶点、活性指标）在同一个数学对象中进行联合表示。通过为不同类型的科学关系（如化学转化、条件调控、分子-靶点相互作用）设计专属的消息传递机制，RAG能够被新一代的异构图神经网络（如我们提出的M²RA-GNN）端到端地学习，从而一体化地捕捉从“如何合成”到“功能如何”的全链条因果关系。本理论为解决多模态科学数据的统一建模难题提供了基础性框架。

1. RAG的形式化定义
反应-活性图（RAG）被形式化地定义为一个带类型映射的异构图：

<img src="https://latex.codecogs.com/svg.latex?G%20%3D%20(V%2C%20E%2C%20%5Ctau%2C%20%5Crho" title="G = (V, E, \tau, \rho)" />

其中：

V 是节点集合。

E 是边集合。

<img src="https://latex.codecogs.com/svg.latex?%5Ctau%3A%20V%20%5Crightarrow%20T_V" title="\tau: V \rightarrow T_V" /> 是一个节点类型映射函数，它将每个节点 <img src="https://latex.codecogs.com/svg.latex?v%20%5Cin%20V" title="v \in V" /> 映射到一个预定义的节点类型。在本项目场景中，节点类型集合 <img src="https://latex.codecogs.com/svg.latex?T_V%20%3D%20%5C%7B%5Ctext%7B%E5%89%8D%E4%BD%93%E5%88%86%E5%AD%90%2C%20%E4%BA%A7%E7%89%A9%E5%88%86%E5%AD%90%2C%20%E5%8F%8D%E5%BA%94%E6%9D%A1%E4%BB%B6%2C%20%E7%94%9F%E7%89%A9%E9%9D%B6%E7%82%B9%2C%20%E6%B4%BB%E6%80%A7%E6%8C%87%E6%A0%87%7D%7D" title="T_V = {\text{前体分子, 产物分子, 反应条件, 生物靶点, 活性指标}}" />。

<img src="https://latex.codecogs.com/svg.latex?%5Crho%3A%20E%20%5Crightarrow%20T_E" title="\rho: E \rightarrow T_E" /> 是一个边类型映射函数，它将每条边 <img src="https://latex.codecogs.com/svg.latex?e%20%5Cin%20E" title="e \in E" /> 映射到一个预定义的关系类型。在本项目场景中，关系类型集合 <img src="https://latex.codecogs.com/svg.latex?T_E%20%3D%20%5C%7B%5Ctext%7B%E5%8C%96%E5%AD%A6%E8%BD%AC%E5%8C%96%2C%20%E6%9D%A1%E4%BB%B6%E8%B0%83%E6%8E%A7%2C%20%E5%88%86%E5%AD%90-%E9%9D%B6%E7%82%B9%E4%BD%9C%E7%94%A8%7D%7D" title="T_E = {\text{化学转化, 条件调控, 分子-靶点作用}}" />。

2. 层次化结构
RAG具有一个内在的双层级结构，使其能够进行多尺度的信息表征：

宏观（概念）层面：图由上述 <img src="https://latex.codecogs.com/svg.latex?T_V" title="T_V" /> 中定义的“概念节点”构成，描述了科学发现的宏观逻辑流程。

微观（原子）层面：每一个“分子”类型的概念节点（无论是前体还是产物），其本身可以被展开为一个由原子和化学键构成的原子图。

3. 核心创新：边类型专属的消息传递机制
作用于RAG之上的图神经网络（如M²RA-GNN）必须采用异构消息传递范式。一个节点 <img src="https://latex.codecogs.com/svg.latex?v" title="v" /> 在第 <img src="https://latex.codecogs.com/svg.latex?k%2B1" title="k+1" /> 层的隐状态 <img src="https://latex.codecogs.com/svg.latex?h_v%5E%7B(k%2B1)%7D" title="h_v^{(k+1)}" /> 的更新逻辑如下：

<img src="https://latex.codecogs.com/svg.latex?h_v%5E%7B(k%2B1)%7D%20%3D%20%5Ctext%7BAGGREGATE%7D%20%5Cleft(%20%5Cleft%5C%7B%20%5Cphi_r(h_v%5E%7B(k)%7D%2C%20h_u%5E%7B(k)%7D)%20%5Cmid%20u%20%5Cin%20%5Cmathcal%7BN%7D_r(v)%2C%20r%20%5Cin%20T_E%20%5Cright%5C%7D%20%5Cright" title="h_v^{(k+1)} = \text{AGGREGATE} \left( \left{ \phi_r(h_v^{(k)}, h_u^{(k)}) \mid u \in \mathcal{N}_r(v), r \in T_E \right} \right)" />

其中 <img src="https://latex.codecogs.com/svg.latex?%5Cphi_r" title="\phi_r" /> 是为关系类型 <img src="https://latex.codecogs.com/svg.latex?r" title="r" /> 设计的专属消息传递函数。例如：

<img src="https://latex.codecogs.com/svg.latex?%5Cphi_%7B%5Ctext%7B%E5%8C%96%E5%AD%A6%E8%BD%AC%E5%8C%96%7D%7D" title="\phi_{\text{化学转化}}" />: 用于建模原子重组与成键断裂规则的函数。

<img src="https://latex.codecogs.com/svg.latex?%5Cphi_%7B%5Ctext%7B%E6%9D%A1%E4%BB%B6%E8%B0%83%E6%8E%A7%7D%7D" title="\phi_{\text{条件调控}}" />: 用于建模温度、溶剂等参数如何影响化学转化过程的函数。

<img src="https://latex.codecogs.com/svg.latex?%5Cphi_%7B%5Ctext%7B%E5%88%86%E5%AD%90-%E9%9D%B6%E7%82%B9%E4%BD%9C%E7%94%A8%7D%7D" title="\phi_{\text{分子-靶点作用}}" />: 用于建模分子三维构象与靶点口袋互补性的函数。

4. M²RA-GNN：RAG理论的实现架构
我们设计的**多模态反应-活性图神经网络（M²RA-GNN）**是RAG理论的具体算法实现。其核心是一个遵循边类型专属消息传递机制的异构图卷积网络，并通过多层次自监督 pre-training 和与分层贝叶斯优化的集成，来解决科研数据稀疏和多目标优化的问题。
