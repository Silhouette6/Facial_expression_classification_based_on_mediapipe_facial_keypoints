"""
(utils)
这个脚本用来show一个npy文件
"""
import numpy as np

# 加载 .npy 文件
file_path0 = "./output/1226.npy"
file_path1 = "datasets/maps/1226.npy"
data = np.load(file_path1)

# 打印数组内容和形状
print("数据形状:", data.shape)
print("数据内容:", data)