"""
分层贝叶斯优化：实现内圈（反应条件）和外圈（分子结构）优化
"""

import numpy as np
from skopt import Optimizer
from skopt.space import Real, Categorical

class HierarchicalBO:
    def __init__(self, model, init_data):
        self.model = model  # M²RA-GNN模型
        self.graph = init_data.get_graph()  # 初始RAG图
        self.outer_bo = self._init_outer_loop()  # 外圈：分子结构优化
        self.inner_bo = self._init_inner_loop()  # 内圈：反应条件优化

    def _init_outer_loop(self):
        """外圈优化：分子结构参数搜索空间"""
        space = [
            Integer(1, 5, name="ring_count"),  # 环数
            Categorical(["-OH", "-NH2", "-COOH"], name="functional_group")  # 官能团
        ]
        return Optimizer(space, base_estimator="gp", acq_func="EI")

    def _init_inner_loop(self):
        """内圈优化：反应条件搜索空间"""
        space = [
            Real(20.0, 150.0, name="temperature"),  # 温度(℃)
            Categorical(["water", "DMSO", "ethanol"], name="solvent")  # 溶剂
        ]
        return Optimizer(space, base_estimator="gp", acq_func="EI")

    def suggest_outer(self, n=1):
        """推荐新分子结构（外圈）"""
        return self.outer_bo.ask(n_points=n)

    def suggest_inner(self, molecule, n=1):
        """为给定分子推荐反应条件（内圈）"""
        # 用模型预测初始化内圈BO
        X_init = np.array([[50.0, "water"], [100.0, "DMSO"]])  # 初始条件
        y_init = -self._evaluate(molecule, X_init)  # 负号用于最小化
        self.inner_bo.tell(X_init.tolist(), y_init.tolist())
        return self.inner_bo.ask(n_points=n)

    def _evaluate(self, molecule, conditions):
        """评估分子-条件组合的综合分数（产率*0.6 + 活性*0.4）"""
        # 此处简化：实际应根据分子和条件更新RAG图并调用模型预测
        scores = []
        for cond in conditions:
            temp, solvent = cond
            # 模拟模型预测（实际需替换为真实推理）
            pred_yield = 0.7 + 0.001 * (temp - 80)  # 示例产率
            pred_activity = 0.8 if solvent == "DMSO" else 0.6  # 示例活性
            scores.append(0.6 * pred_yield + 0.4 * pred_activity)
        return np.array(scores)

    def update(self, X, y):
        """用实验结果更新BO模型"""
        self.outer_bo.tell(X, [-val for val in y])  # 负号转换
