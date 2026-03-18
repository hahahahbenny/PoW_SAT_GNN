import torch
import torch.nn.functional as F
from sklearn.metrics import ndcg_score
from scipy.stats import pearsonr, spearmanr
import numpy as np

def compute_metrics(pred, target, k_list=[100, 1000]):
    """
    一次性计算 NDCG, Pearson, Spearman
    """
    y_true = target.detach().cpu().numpy()
    y_score = pred.detach().cpu().numpy()
    
    # 1. 计算不同尺度的 NDCG
    results = {}
    for k in k_list:
        results[f'ndcg_{k}'] = ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=k)
    
    # 2. 相关系数 (衡量整体拟合)
    results['pearson'], _ = pearsonr(y_score, y_true)
    results['spearman'], _ = spearmanr(y_score, y_true)
    
    # 3. Top-K 命中率 (针对决策堆顶端的分析)
    top_true = np.argsort(y_true)[-k_list[0]:]
    top_pred = np.argsort(y_score)[-k_list[0]:]
    results['top_k_hit'] = len(set(top_true) & set(top_pred)) / k_list[0]
    
    return results

def train_epoch(model, loader, optimizer, device, margin):
    model.train()
    total_loss = 0
    grad_norms = []
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = ranking_loss(out.squeeze(), data['variable'].y, margin=margin)
        loss.backward()
        
        # 记录梯度范数 (监控模型健康状况)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norms.append(grad_norm.item())
        
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader), np.mean(grad_norms)

def validate(model, loader, device):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x_dict, data.edge_index_dict)
            metrics = compute_metrics(out.squeeze(), data['variable'].y)
            all_metrics.append(metrics)
            
    # 对所有测试用例取平均值
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    return avg_metrics