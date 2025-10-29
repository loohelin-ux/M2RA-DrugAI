"""分层贝叶斯优化器（双重闭环）"""
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

class HierarchicalBO:
    def __init__(self, model, search_space, device="cpu"):
        self.model = model.eval()
        self.device = device
        self.mol_space = search_space["molecules"]
        self.cond_space = search_space["conditions"]
        # 初始化内循环BO模型
        self.inner_bo = SingleTaskGP(
            train_X=torch.tensor([[30.0]], device=device),
            train_Y=torch.tensor([[0.5]], device=device)
        )

    def _score_molecule(self, mol_smiles):
        """评估分子活性-合成可行性综合得分"""
        # 模拟模型预测
        dummy_rag = self._build_dummy_rag(mol_smiles)
        with torch.no_grad():
            pred = self.model(dummy_rag)
        activity_score = pred["activity"].item()
        # 模拟合成可行性（分子越短越易合成）
        synth_feasibility = 1.0 - (len(mol_smiles) / 100)
        return 0.7 * activity_score + 0.3 * synth_feasibility

    def _optimize_conditions(self, mol_smiles):
        """内循环：优化反应条件"""
        bounds = torch.tensor([[self.cond_space["temperature"][0]], 
                               [self.cond_space["temperature"][1]]], device=self.device)
        acq_func = ExpectedImprovement(self.inner_bo, best_f=self.inner_bo.train_Y.max())
        optimal_cond, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20
        )
        # 随机选择溶剂
        solvents = self.cond_space["solvent"]
        optimal_solvent = solvents[torch.randint(0, len(solvents), (1,)).item()]
        return {"temperature": round(optimal_cond.item(), 1), "solvent": optimal_solvent}

    def suggest(self, n_suggestions=1):
        """外循环：推荐最优分子+条件组合"""
        top_mols = sorted(self.mol_space, key=lambda x: self._score_molecule(x), reverse=True)[:n_suggestions]
        suggestions = []
        for mol in top_mols:
            optimal_cond = self._optimize_conditions(mol)
            suggestions.append({"molecule": mol, "conditions": optimal_cond})
        return suggestions

    def _build_dummy_rag(self, mol_smiles):
        """构建虚拟RAG图（用于预测）"""
        from m2ra_drugai.data.builder import RAGDataBuilder
        dummy_sample = (mol_smiles, mol_smiles, {"temperature": 30}, "dummy_target", 0.0)
        return RAGDataBuilder().build([dummy_sample])
