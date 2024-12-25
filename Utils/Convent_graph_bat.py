"""
(utils)
这个脚本用来批量将4x478的npy文件构建pkl图文件
"""
import pickle
import os
import numpy as np
import torch
import mediapipe as mp
from torch_geometric.data import Data

# 初始化 Mediapipe 的人脸模型
mp_face_mesh = mp.solutions.face_mesh
connections = mp_face_mesh.FACEMESH_TESSELATION  # 人脸拓扑结构

# 获取 ./datasets/maps/ 下所有的 .npy 文件
input_dir = "./datasets/maps/"
output_dir = "./datasets/graphs/"
os.makedirs(output_dir, exist_ok=True)  # 创建保存图数据的文件夹（如果不存在）

count = 0
# 遍历所有 .npy 文件
for npy_file in os.listdir(input_dir):
    count += 1
    if npy_file.endswith(".npy"):
        # 读取 .npy 文件
        face_4x478 = np.load(os.path.join(input_dir, npy_file))  # 形状: (4, 478)

        # 提取前三行作为 3D 坐标
        face_3x478 = face_4x478[:3, :]

        face_coordinates = face_3x478.T  # 转置为 (478, 3)

        # 节点特征 (Node Features)
        x = torch.tensor(face_coordinates, dtype=torch.float)

        # 将 edges 转为 edge_index
        edges = list(connections)  # 连接关系
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # 转置为 (2, E)

        # 标签 (是否微笑：0 或 1)，假设标签存储在第 4 行的第 1 列
        label = torch.tensor([int(face_4x478[3, 0])], dtype=torch.long)

        # 构建 PyTorch Geometric 图数据
        graph_data = Data(x=x, edge_index=edge_index, y=label)

        # 保存图数据
        graph_file = os.path.join(output_dir, f"{npy_file.split('.')[0]}.pkl")
        with open(graph_file, "wb") as f:
            pickle.dump(graph_data, f)

        print(f"图数据已保存：{graph_file}",count)

print("所有图数据已处理完成！")
