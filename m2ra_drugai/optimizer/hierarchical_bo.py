"""
Hierarchical Bayesian Optimization (BO) for Dual-Loop Active Learning.
"""

import numpy as np
from skopt import Optimizer as SKOptimizer
from skopt.space import Real, Categorical, Integer
from m2ra_drugai.data.builder import RAGData

class HierarchicalBO:
    def __init__(self, model, search_space, initial_data: RAGData):
        self.model = model  # M²RA-GNN模型（作为代理模型）
        self.search_space = self._parse_search_space(search_space)
        self.initial_data = initial_data
        
        # 初始化贝叶斯优化器（使用skopt作为基础）
        self.bo = SKOptimizer(
            dimensions=self.search_space,
            base_estimator="gp",  # 高斯过程作为代理模型
            acq_func="EI"  # 期望改进 acquisition function
        )
        
        # 用初始数据预热优化器
        self._warmup_bo()

    def _parse_search_space(self, space_dict):
        """将用户定义的搜索空间转换为skopt格式"""
        parsed = []
        for key, val in space_dict.items():
            if isinstance(val, tuple) and len(val) == 2 and all(isinstance(x, float) for x in val):
                parsed.append(Real(val[0], val[1], name=key))
            elif isinstance(val, list) and all(isinstance(x, str) for x in val):
                parsed.append(Categorical(val, name=key))
            elif isinstance(val, tuple) and len(val) == 2 and all(isinstance(x, int) for x in val):
                parsed.append(Integer(val[0], val[1], name=key))
            else:
                raise ValueError(f"不支持的搜索空间格式: {key}={val}")
        return parsed

    def _warmup_bo(self):
        """用初始数据训练代理模型"""
        # 提取初始数据的特征和目标值（简化示例）
        X_initial = self.initial_data.get_search_space_samples()  # 需要在RAGData中实现
        y_initial = - (  # 负号因为skopt默认最小化目标
            0.6 * self.initial_data.yield_labels +  # 产率权重0.6
            0.4 * self.initial_data.activity_labels  # 活性权重0.4
        )
        self.bo.tell(X_initial, y_initial)

    def _objective_function(self, params):
        """目标函数：调用M²RA-GNN预测并计算综合分数"""
        # 将参数转换为字典（带名称）
        param_dict = {dim.name: param for dim, param in zip(self.search_space, params)}
        
        # 1. 内圈优化：固定分子，优化反应条件（预测产率）
        yield_pred = self.model.predict_yield(param_dict)
        
        # 2. 外圈优化：基于条件优化分子结构（预测活性）
        activity_pred = self.model.predict_activity(param_dict)
        
        # 多目标综合分数（产率*0.6 + 活性*0.4，越高越好）
        return - (0.6 * yield_pred + 0.4 * activity_pred)  # 负号用于最小化

    def suggest(self, n_suggestions=1):
        """生成下一批实验建议"""
        # 从BO获取候选参数
        candidates = self.bo.ask(n_points=n_suggestions)
        
        # 转换为带名称的字典格式
        suggestions = []
        for cand in candidates:
            sugg_dict = {dim.name: cand[i] for i, dim in enumerate(self.search_space)}
            suggestions.append(sugg_dict)
        
        return suggestions
