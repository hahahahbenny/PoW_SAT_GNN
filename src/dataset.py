import os
import torch
from torch.utils.data import Dataset
import glob

class SATSolverDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: data/processed/ 所在的路径
        """
        self.root_dir = root_dir
        # 查找所有的 .pt 文件
        self.file_list = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        
        if len(self.file_list) == 0:
            print(f"Warning: No .pt files found in {root_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 直接加载预处理好的 HeteroData 对象
        data = torch.load(self.file_list[idx])
        return data

def get_data_list(root_dir):
    """
    辅助函数：一次性加载所有数据到内存（如果内存够大，700个文件约5-8GB，3080Ti机器通常有32GB+内存可用）
    """
    files = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
    data_list = [torch.load(f) for f in files]
    return data_list