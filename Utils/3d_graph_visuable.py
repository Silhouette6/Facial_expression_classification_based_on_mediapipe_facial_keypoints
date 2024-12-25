"""
(utils)
这个脚本用来实装load一个4x478的npy，通过mediapipe解析内容、并构建为一个图(包含标签)。然后用plt 3d可视化构建出来的图
"""

import matplotlib.pyplot as plt
from torch_geometric.data import Data
import numpy as np
import mediapipe as mp
import torch


# 假设你已经提取了人脸关键点的 3D 坐标和边
face_4x478 = np.load("../datasets/npy4x478/1225.npy")  # 形状: (4, 478)
face_3x478 = face_4x478[:3, :]

face_coordinates = face_3x478.T  # 转置为 (478, 3)

# 节点特征 (Node Features)
x = torch.tensor(face_coordinates, dtype=torch.float)

# 边索引 (Edge Index)：这里用 Mediapipe 提供的拓扑结构
# 初始化 Mediapipe 的人脸模型
mp_face_mesh = mp.solutions.face_mesh
connections = mp_face_mesh.FACEMESH_TESSELATION  # 人脸拓扑结构

# 提取连接关系 (edges)
edges = list(connections)  # connections 是一个集合，每个元素是 (start, end)

# 将 edges 转为 edge_index
# 假设 edge_index 已经定义为一个 (2, E) 张量
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # 转置为 (2, E)

# 标签 (是否微笑：0 或 1)
label = torch.tensor([int(face_4x478[3, 0])], dtype=torch.long)

# 构建 PyTorch Geometric 图数据
graph_data = Data(x=x, edge_index=edge_index, y=label)


# 提取节点坐标 (x, y, z)
node_coords = graph_data.x.numpy()  # 转为 NumPy 数组
edge_index = graph_data.edge_index.numpy()  # 转为 NumPy 数组

# 创建一个 3D 图形
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制节点
ax.scatter(
    node_coords[:, 0],  # x 坐标
    node_coords[:, 1],  # y 坐标
    node_coords[:, 2],  # z 坐标
    c='blue',           # 节点颜色
    s=20,               # 节点大小
    label='Nodes'
)

# 绘制边
for start, end in edge_index.T:  # 遍历每条边
    x_coords = [node_coords[start, 0], node_coords[end, 0]]
    y_coords = [node_coords[start, 1], node_coords[end, 1]]
    z_coords = [node_coords[start, 2], node_coords[end, 2]]

    ax.plot(
        x_coords, y_coords, z_coords, c='gray', alpha=0.5, linewidth=0.5
    )

# 添加标题和图例
ax.set_title("3D Graph Visualization", fontsize=16)
ax.legend()
plt.show()