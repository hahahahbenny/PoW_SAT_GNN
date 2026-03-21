import torch
import os
import glob
from tqdm import tqdm

def check_files(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.pt"))
    print(f"开始检查 {len(files)} 个文件...")
    
    bad_files = []
    for f in tqdm(files):
        try:
            # 尝试加载文件头部
            _ = torch.load(f, map_location='cpu')
        except Exception as e:
            print(f"\n损坏文件: {f} | 错误: {e}")
            bad_files.append(f)
    
    if bad_files:
        print(f"\n检查完毕！共发现 {len(bad_files)} 个损坏文件。")
        confirm = input("是否直接删除这些文件？(y/n): ")
        if confirm.lower() == 'y':
            for f in bad_files:
                os.remove(f)
            print("损坏文件已清理。")
    else:
        print("所有文件完整，没有发现损坏。")

if __name__ == "__main__":
    check_files("data/processed")