"""
(utils)
这个脚本用来检查导出的pkl文件是否损坏
"""
import pickle

# 假设你要检查保存的图数据
graph_file = "./datasets/graphs/1225.pkl"  # 修改为你实际的文件路径

# 加载 .pkl 文件
with open(graph_file, "rb") as f:
    loaded_graph_data = pickle.load(f)

# 检查加载后的图数据
print("节点特征 (x):", loaded_graph_data.x)
print("边索引 (edge_index):", loaded_graph_data.edge_index)
print("标签 (y):", loaded_graph_data.y)
