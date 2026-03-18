import pandas as pd
import matplotlib.pyplot as plt

# ... 前面的代码保持不变 ...

def save_plots(history, model_name):
    epochs = range(1, len(history['loss']) + 1)
    
    # 图 1: Loss 与 NDCG 双轴图 (论文核心图表)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Ranking Loss', color=color)
    ax1.plot(epochs, history['loss'], color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('NDCG@1000', color=color)
    ax2.plot(epochs, history['ndcg_1000'], color=color, label='Val NDCG@1000')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'Training Convergence: {model_name}')
    fig.tight_layout()
    plt.savefig(f"result/logs/{model_name}_convergence.png", dpi=300)
    
    # 图 2: 相关性指标趋势
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['pearson'], label='Pearson Corr')
    plt.plot(epochs, history['spearman'], label='Spearman Corr')
    plt.plot(epochs, history['top_k_hit'], label='Top-100 Hit Rate')
    plt.title('Ranking Quality Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(f"result/logs/{model_name}_metrics.png", dpi=300)

# 在 main 循环中：
history = {'loss': [], 'ndcg_1000': [], 'pearson': [], 'spearman': [], 'top_k_hit': [], 'grad_norm': []}

for epoch in range(1, args.epochs + 1):
    loss, g_norm = train_epoch(model, train_loader, optimizer, device, args.margin)
    metrics = validate(model, val_loader, device)
    
    # 记录数据
    history['loss'].append(loss)
    history['grad_norm'].append(g_norm)
    for k, v in metrics.items():
        if k in history: history[k].append(v)
    
    # 打印和保存
    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | NDCG@1000: {metrics['ndcg_1000']:.4f} | Pearson: {metrics['pearson']:.4f}")
    
    # 每 10 轮保存一次 CSV，防止断电丢失
    pd.DataFrame(history).to_csv(f"result/logs/{args.model_name}_raw.csv", index=False)

# 训练结束后画图
save_plots(history, args.model_name)