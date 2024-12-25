import os
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# --- 超参数配置 ---
config = {
    'input_dim': 3,  # 输入特征维度（3D 坐标）
    'hidden_dim': 256,  # 隐藏层的维度
    'output_dim': 2,  # 输出维度（2：微笑与否）
    'batch_size': 32,  # 每个批次的数据量
    'save_dir': './saved_models',  # 模型保存路径文件夹
    'model_name': 'best1.pth'  # 模型文件名
}


# --- 自定义加载数据函数 ---
def load_graph_data_from_directory(directory):
    graph_data_list = []
    for graph_file in os.listdir(directory):
        if graph_file.endswith(".pkl"):
            with open(os.path.join(directory, graph_file), "rb") as f:
                graph_data = pickle.load(f)
                graph_data_list.append(graph_data)
    return graph_data_list


# --- 定义 GCN 模型 ---
class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)  # 输出一个图的标签

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x


# --- 测试函数 ---
def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 推理时不需要计算梯度
        for data in loader:
            out = model(data)  # 模型输出 logits
            pred = out.argmax(dim=1)  # 获取预测类别

            # 打印每个样本的预测值与真实值
            for i in range(len(data.y)):
                print(f"Sample {i + 1}: Predicted: {pred[i].item()}, Actual: {data.y[i].item()}")

            correct += (pred == data.y).sum().item()  # 与真实标签比较
            total += data.y.size(0)  # 累加样本总数
    return correct / total  # 返回准确率


if __name__ == '__main__':
    # 加载测试数据
    test_data = load_graph_data_from_directory("./datasets/graphs/test")
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

    # 实例化模型并加载训练好的权重
    model = GCNModel(input_dim=config['input_dim'], hidden_dim=config['hidden_dim'], output_dim=config['output_dim'])

    # 假设你想加载最后一个保存的模型
    model_path = os.path.join(config['save_dir'], config['model_name'])  # 修改为你保存的模型文件
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置模型为评估模式

    # 测试模型并打印每个样本的预测值和真实值
    accuracy = test(model, test_loader)
    print(f"Test Accuracy: {accuracy:.4f}")

