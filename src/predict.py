import os
import torch
import numpy as np
import pandas as pd
import time
from torch_geometric.data import HeteroData
from model import HGCNIISolver

# 性能优化：开启 CPU 多线程并行
# torch.set_num_threads(os.cpu_count())

# ==========================================
# 1. 极速 CNF 解析逻辑 (Numpy 向量化)
# ==========================================
def fast_cnf_to_data(cnf_path, mapping):
    """利用 Numpy 向量化操作加速解析 20w 行级别的 CNF"""
    start_time = time.time()
    
    with open(cnf_path, 'r') as f:
        all_lines = f.readlines()

    # 1. 提取元数据与正文
    max_var = 0
    num_clauses = 0
    body_lines = []
    
    for line in all_lines:
        if line.startswith('c'): continue
        if line.startswith('p'):
            parts = line.split()
            max_var = int(parts[2])
            num_clauses = int(parts[3])
            continue
        if line.strip():
            body_lines.append(line)

    # 2. 将所有子句展平为长整型数组
    # 使用 join 拼接字符串并一次性转换为 numpy 数组，避开 Python 循环
    raw_data = np.fromstring(" ".join(body_lines), dtype=int, sep=' ')
    
    # 3. 识别子句边界 (0 是 DIMACS 的分隔符)
    zero_indices = np.where(raw_data == 0)[0]
    # 计算每个子句的长度 (不计最后的0)
    clause_lengths = np.diff(np.hstack([-1, zero_indices])) - 1
    
    # 4. 提取文字并过滤 0
    lits = raw_data[raw_data != 0]
    
    # 5. 构建边关系索引 (Variable <-> Clause)
    # 为每个文字生成对应的子句 ID
    clause_ids = np.repeat(np.arange(num_clauses), clause_lengths)
    var_ids_0based = np.abs(lits) - 1
    
    pos_mask = (lits > 0)
    neg_mask = ~pos_mask
    
    # 6. 计算变量度数 (用于特征 f4)
    var_degrees = np.bincount(var_ids_0based, minlength=max_var)
    
    # 7. 识别固定位 (长度为 1 的子句)
    unit_clause_mask = (clause_lengths == 1)
    # 仅提取长度为 1 的子句中的文字
    unit_lits = lits[np.repeat(unit_clause_mask, clause_lengths)]
    unit_vars_idx = np.abs(unit_lits) - 1

    # 8. 构建特征矩阵 [f1, f2, f3, f4]
    v_feat = torch.zeros((max_var, 4))
    
    # f1: Nonce (1-32)
    v_feat[0:32, 0] = 1.0
    
    # f2: Fixed (单子句)
    if len(unit_vars_idx) > 0:
        v_feat[unit_vars_idx, 1] = 1.0
        
    # f3: Crypto Logic (根据映射表范围批量打标)
    # 优化：利用 tensor 范围切片
    all_crypto_intervals = list(mapping.values())
    for start, end in all_crypto_intervals:
        # 注意 mapping 是 1-based，转换为 0-based 切片
        idx_s, idx_e = max(0, start-1), min(max_var, end)
        v_feat[idx_s:idx_e, 2] = 1.0
        
    # f4: 度数归一化
    v_feat[:, 3] = torch.from_numpy(var_degrees).float() / (num_clauses + 1e-9)

    # 9. 封装 HeteroData
    data = HeteroData()
    data['variable'].x = v_feat
    data['clause'].x = torch.ones((num_clauses, 1))
    
    # 建立边索引
    data['variable', 'pos_in', 'clause'].edge_index = torch.stack([
        torch.from_numpy(var_ids_0based[pos_mask]), 
        torch.from_numpy(clause_ids[pos_mask])
    ]).long()
    
    data['variable', 'neg_in', 'clause'].edge_index = torch.stack([
        torch.from_numpy(var_ids_0based[neg_mask]), 
        torch.from_numpy(clause_ids[neg_mask])
    ]).long()
    
    # 补全反向边
    data['clause', 'rev_pos_in', 'variable'].edge_index = data['variable', 'pos_in', 'clause'].edge_index.flip(0)
    data['clause', 'rev_neg_in', 'variable'].edge_index = data['variable', 'neg_in', 'clause'].edge_index.flip(0)
    
    print(f"CNF 解析与图构建耗时: {time.time() - start_time:.2f}s")
    return data

# ==========================================
# 2. 分段映射函数 (肩部+悬崖)
# ==========================================
def get_difficulty_anchored_score(pred_scores, target_bits, pivot_k=8000):
    num_vars = pred_scores.size(0)
    
    # 难度天花板统计
    stats_map = {10: 2000, 11: 3500, 12: 6000, 13: 9000, 14: 20000, 15: 50000, 16: 120000}
    target_max = stats_map.get(target_bits, 120000)

    # 计算排名
    _, sorted_indices = torch.sort(pred_scores, descending=True)
    ranks = torch.zeros(num_vars, dtype=torch.long)
    ranks[sorted_indices] = torch.arange(num_vars)

    final_scores = torch.zeros(num_vars, dtype=torch.float32)

    # A: 肩部 (线性缓慢下降)
    top_mask = (ranks <= pivot_k)
    norm_top_rank = ranks[top_mask].float() / pivot_k
    final_scores[top_mask] = target_max * (1.0 - 0.2 * norm_top_rank)

    # B: 悬崖 (指数级快速坠落)
    bottom_mask = ~top_mask
    dist_from_pivot = (ranks[bottom_mask] - pivot_k).float()
    final_scores[bottom_mask] = (target_max * 0.8) * torch.exp(-0.001 * dist_from_pivot)

    return final_scores.numpy()

# ==========================================
# 3. 推理主程序
# ==========================================
def run_prediction(cnf_path, model_path, mapping_path, base_output_path, target_bits):
    # 1. 自动检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在分析 CNF 并执行推理: {os.path.basename(cnf_path)}")
    print(f"设定目标难度: {target_bits}-bit | 使用设备: {device}")

    # 2. 加载映射表
    df_map = pd.read_csv(mapping_path)
    mapping = {row['VarName']: (int(row['Start']), int(row['End'])) for _, row in df_map.iterrows()}

    # 3. 极速转图 (此时 data 在 CPU 上)
    data = fast_cnf_to_data(cnf_path, mapping)
    
    # 【核心修复点 A】：显式将整个图对象搬到 GPU/CPU
    data = data.to(device)

    # 4. 模型加载
    model = HGCNIISolver(hidden_channels=64, num_layers=8)
    
    # 【核心修复点 B】：加载权重时指定 map_location，并搬运模型
    # 这样可以兼容“GPU训练的模型在CPU上跑”或“GPU上跑”
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device) # 确保模型所有参数在对应设备上
    model.eval()

    # 5. 执行推理
    start_inf = time.time()
    with torch.inference_mode():
        # 这里传入的 data.x_dict 已经是搬到 device 之后的了
        raw_output = model(data.x_dict, data.edge_index_dict).squeeze()
        
        # 将结果转回 CPU 以后再进行分段映射（因为后续涉及 numpy 操作）
        final_scores = get_difficulty_anchored_score(raw_output.cpu(), target_bits)
        
    print(f"GNN 推理与分值映射耗时: {time.time() - start_inf:.2f}s")

    # 5. 导出文本格式 (VarID Score)
    txt_path = base_output_path + ".weights"
    with open(txt_path, 'w') as f:
        for i, s in enumerate(final_scores):
            f.write(f"{i+1} {s:.6f}\n")

    # 6. 导出二进制格式 (C double 数组)
    bin_path = base_output_path + ".bin"
    final_scores.astype(np.float64).tofile(bin_path)
    
    print(f"总流程处理完成。二进制权重存至: {bin_path}")

if __name__ == "__main__":
    # --- 路径配置 ---
    TARGET_CNF = "/home/ubuntu/gnn_for_sat/PoW_SAT_GNN/data/cnf/14bits_left.cnf"
    MODEL_FILE = "/home/ubuntu/gnn_for_sat/PoW_SAT_GNN/result/check_points/hgcnii_full_best.pth"
    MAP_FILE = "/home/ubuntu/gnn_for_sat/PoW_SAT_GNN/data/map/global_variable_mapping.csv"
    
    SAVE_DIR = "/home/ubuntu/gnn_for_sat/PoW_SAT_GNN/result/weights_out"
    os.makedirs(SAVE_DIR, exist_ok=True)
    OUTPUT_BASE = os.path.join(SAVE_DIR, os.path.basename(TARGET_CNF).replace(".cnf", ""))

    run_prediction(TARGET_CNF, MODEL_FILE, MAP_FILE, OUTPUT_BASE, target_bits=14)