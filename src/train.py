import torch
import torch.nn.functional as F
from sklearn.metrics import ndcg_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
import time

def ranking_loss(pred, target, margin=0.2, num_pairs=4000):
    n = pred.size(0)
    idx_i = torch.randint(0, n, (num_pairs,))
    idx_j = torch.randint(0, n, (num_pairs,))
    s_i, s_j = pred[idx_i], pred[idx_j]
    y_i, y_j = target[idx_i], target[idx_j]
    label = torch.sign(y_i - y_j)
    mask = (label != 0)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)
    return F.margin_ranking_loss(s_i[mask], s_j[mask], label[mask], margin=margin)

def compute_metrics(pred, target, k_list=[100, 500, 1000]):
    y_true = target.detach().cpu().numpy()
    y_score = pred.detach().cpu().numpy()
    results = {}
    
    # NDCG 计算
    y_true_reshaped = y_true.reshape(1, -1)
    y_score_reshaped = y_score.reshape(1, -1)
    for k in k_list:
        results[f'ndcg_{k}'] = ndcg_score(y_true_reshaped, y_score_reshaped, k=k)
    
    # 相关性计算 (增加 epsilon 防止常数标签导致 NaN)
    if np.std(y_true) > 1e-9 and np.std(y_score) > 1e-9:
        results['pearson'], _ = pearsonr(y_score, y_true)
        results['spearman'], _ = spearmanr(y_score, y_true)
    else:
        results['pearson'], results['spearman'] = 0, 0

    # Hit Rate 计算
    true_top_indices = np.argsort(y_true)
    pred_top_indices = np.argsort(y_score)
    for k in k_list:
        actual_top_k = set(true_top_indices[-k:])
        predicted_top_k = set(pred_top_indices[-k:])
        results[f'hit_{k}'] = len(actual_top_k & predicted_top_k) / float(k)
    
    return results

def train_epoch(model, loader, optimizer, device, margin):
    model.train()
    total_loss = 0
    grad_norms = []
    total_samples = len(loader)
    start_time = time.time()

    for i, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = ranking_loss(out.squeeze(), data['variable'].y, margin=margin)
        loss.backward()
        g_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norms.append(g_norm.item())
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"    [Train] {i+1}/{total_samples} | Loss: {loss.item():.4f} | G-Norm: {g_norm.item():.2f} | {elapsed/(i+1):.2f}s/it", flush=True)

    return total_loss / total_samples, np.mean(grad_norms)

def validate(model, loader, device):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x_dict, data.edge_index_dict)
            all_metrics.append(compute_metrics(out.squeeze(), data['variable'].y))
            
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    return avg_metrics