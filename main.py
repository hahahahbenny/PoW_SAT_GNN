import argparse, os, time, torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from src.model import HGCNIISolver
from src.dataset import SATSolverDataset
from src.train import train_epoch, validate

import matplotlib
matplotlib.use('Agg') # 必须加在 import pyplot 之前！防止远程服务器报错
import matplotlib.pyplot as plt

def save_plots(history, model_name):
    epochs = range(1, len(history['loss']) + 1)
    log_dir = "result/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置全局学术风格
    plt.style.use('seaborn-v0_8-paper') 

    # --- 图 1: 训练收敛图 (Ranking Loss vs NDCG@1000) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, history['loss'], 'r-', label='Train Ranking Loss', linewidth=1.5)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss (Margin Ranking)', color='r', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(epochs, history['ndcg_1000'], 'b-', label='Val NDCG@1000', linewidth=2)
    ax2.set_ylabel('NDCG @ Top 1000', color='b', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='b')
    
    plt.title(f"Model Convergence Analysis: {model_name}", fontsize=14)
    fig.tight_layout()
    plt.savefig(f"{log_dir}/{model_name}_convergence.png", dpi=300)
    plt.close()

    # --- 图 2: 关键变量命中率分析 (Hit Rate @ 100/500/1000) ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['hit_100'], label='Hit Rate @ 100', alpha=0.7)
    plt.plot(epochs, history['hit_500'], label='Hit Rate @ 500', alpha=0.8)
    plt.plot(epochs, history['hit_1000'], label='Hit Rate @ 1000', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Hit Rate (Overlap with Ground Truth)', fontsize=12)
    plt.title(f"Backdoor Variable Identification Accuracy: {model_name}", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{log_dir}/{model_name}_hit_rates.png", dpi=300)
    plt.close()

    # --- 图 3: 排序一致性指标 (Pearson & Spearman) ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['pearson'], label='Pearson Correlation', linestyle='--')
    plt.plot(epochs, history['spearman'], label='Spearman Correlation', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Correlation Coefficient', fontsize=12)
    plt.title(f"Global Ranking Consistency: {model_name}", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{log_dir}/{model_name}_correlations.png", dpi=300)
    plt.close()

    print(f"--- 所有高清实验图表已保存至 {log_dir} ---")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--no_res', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs("result/check_points", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = SATSolverDataset(root_dir="data/processed")
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=1, num_workers=args.num_workers)

    model = HGCNIISolver(hidden_channels=64, use_inter_layer_res=not args.no_res).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 自动根据验证集返回的 key 初始化 history
    history = {}
    best_ndcg = 0

    for epoch in range(1, args.epochs + 1):
        loss, g_norm = train_epoch(model, train_loader, optimizer, device, margin=0.2)
        metrics = validate(model, val_loader, device)
        
        # 记录 Loss 和 GradNorm
        history.setdefault('loss', []).append(loss)
        history.setdefault('grad_norm', []).append(g_norm)
        
        # 自动记录 metrics 里的所有项 (ndcg_100, hit_100 etc.)
        for k, v in metrics.items():
            history.setdefault(k, []).append(v)
            
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | NDCG@1000: {metrics['ndcg_1000']:.4f} | Hit@100: {metrics['hit_100']:.2f}")
        
        if epoch % 5 == 0:
            pd.DataFrame(history).to_csv(f"result/logs/{args.model_name}_raw.csv", index=False)
            
        if metrics['ndcg_1000'] > best_ndcg:
            best_ndcg = metrics['ndcg_1000']
            torch.save(model.state_dict(), f"result/check_points/{args.model_name}_best.pth")
        
        scheduler.step()

    save_plots(history, args.model_name)
    torch.save(model.state_dict(), f"result/check_points/{args.model_name}_final.pth")

if __name__ == "__main__":
    main()