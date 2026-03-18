import torch
import os
from model import SHA256SolverGNN # 引用你最开始给出的结构

def generate_kissat_weights(cnf_path, model_path, output_weights_path):
    device = torch.device('cpu') # 推理通常CPU够快
    
    # 1. 重新构建图 (逻辑同 preprocess.py, 但不读取 label)
    # ... (此处省略重复的 parse_cnf_to_lcg 逻辑)
    # 假设得到了 data 对象
    
    model = SHA256SolverGNN(hidden_channels=64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        scores = model(data.x_dict, data.edge_index_dict)
        scores = scores.squeeze().numpy()
    
    # 2. 写入 Kissat 格式文件：每行一个变量的分数
    # 格式：VarID Score
    with open(output_weights_path, 'w') as f:
        for i, s in enumerate(scores):
            f.write(f"{i+1} {s:.6f}\n")
    print(f"权重文件已生成至：{output_weights_path}")

# 使用示例
# generate_kissat_weights("test_24R.cnf", "result/best_model.pth", "result/weights_out/test_24R.weights")