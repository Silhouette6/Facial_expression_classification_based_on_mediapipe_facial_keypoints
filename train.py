import os
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.optim.lr_scheduler import LambdaLR

# --- 超参数配置 ---
config = {
    'input_dim': 3,  # 输入特征维度（3D 坐标）
    'hidden_dim': 256,  # 隐藏层的维度
    'output_dim': 2,  # 输出维度（2：微笑与否）
    'learning_rate': 0.002,  # 初始学习率
    'batch_size': 32,  # 每个批次的数据量
    'epochs': 400,  # 训练的轮数
    'save_interval': 5,  # 每几个epoch保存一次模型
    'print_every_sample': 5,  # 每几个样本汇报一次效果
    'save_dir': './saved_models',  # 模型保存路径
    'log_file': './training_log.txt',  # 训练日志文件路径
    'threshold': 0.5,  # 判断为正样本的置信度
    'lr_stable_epochs': 360,  # 固定学习率的轮数
    'lr_decay_epochs': 240,  # 学习率衰减的轮数
}

# 确保保存目录存在
os.makedirs(config['save_dir'], exist_ok=True)

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
        # 图卷积操作
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)  # 新增加的卷积层
        x = F.relu(x)

        # 使用池化操作聚合节点特征为图级别的表示
        x = global_mean_pool(x, data.batch)  # 将每个图的节点特征聚合成一个图级别的表示

        # 通过全连接层输出图的分类结果
        x = self.fc(x)
        return x  # 这里输出的是图级别的预测


# --- 加载数据 ---
train_data = load_graph_data_from_directory("./datasets/graphs/train")
test_data = load_graph_data_from_directory("./datasets/graphs/test")
train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

# --- 实例化模型 ---
model = GCNModel(input_dim=config['input_dim'], hidden_dim=config['hidden_dim'], output_dim=config['output_dim'])

# --- 训练函数 ---
criterion = torch.nn.CrossEntropyLoss()  # 使用 CrossEntropyLoss，因为输出是两个类别的 logits
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# --- 学习率调度器 ---
def lr_lambda(epoch, stable_epochs, decay_epochs):
    # 在前 stable_epochs 轮次，学习率保持不变
    if epoch < stable_epochs:
        return 1.0
    # 在接下来的 decay_epochs 轮次，逐步递减学习率
    elif epoch < stable_epochs + decay_epochs:
        return 1.0 - (epoch - stable_epochs) / decay_epochs
    else:
        return 0.0  # 学习率衰减到 0

scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr_lambda(epoch, config['lr_stable_epochs'], config['lr_decay_epochs']))

# 获取当前学习率
def get_current_lr(optimizer):
    # 假设只有一个参数组，获取第一个参数组的学习率
    return optimizer.param_groups[0]['lr']

# 训练日志函数
def log_training_info(epoch, loss, accuracy, lr, log_file=config['log_file']):
    with open(log_file, 'a') as f:
        f.write(f"Epoch {epoch},Loss: {loss:.4f},Accuracy: {accuracy:.4f},lr: {lr}\n")

def train(model, loader, optimizer, criterion, scheduler):
    model.train()  # 设置模型为训练模式
    total_loss = 0  # 记录总损失
    correct = 0  # 记录正确预测的样本数
    total = 0  # 记录总样本数

    for batch_idx, data in enumerate(loader):  # 遍历加载器中的每个批次
        optimizer.zero_grad()  # 每次更新参数之前清空梯度

        # 将数据送入模型，得到预测输出
        out = model(data)

        # 计算损失
        loss = criterion(out, data.y)  # CrossEntropyLoss自动计算softmax
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 累积损失和正确预测的数量
        total_loss += loss.item()
        pred = out.argmax(dim=1)  # 使用 argmax 获取预测类别
        correct += (pred == data.y).sum().item()  # 与真实标签比较
        total += data.y.size(0)

        # 每5个batch打印一次预测和真实标签
        if batch_idx % config['print_every_sample'] == 0:  # 控制打印频率
            print(f"Batch {batch_idx + 1}: 预测标签: {pred.tolist()}, 真实标签: {data.y.tolist()}")

    # 计算平均损失和准确率
    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    # 打印总损失和准确率
    print(f"训练集损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}")

    # 保存训练日志
    log_training_info(epoch = epoch,loss = avg_loss, accuracy = accuracy, lr = get_current_lr(optimizer))

    # 更新学习率
    scheduler.step()

    return avg_loss, accuracy


# --- 测试函数 ---
def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 推理时不需要计算梯度
        for data in loader:
            out = model(data)  # 模型输出 logits

            # 通过 argmax 获取预测类别
            pred = out.argmax(dim=1)  # 类别 0 或 类别 1

            correct += (pred == data.y).sum().item()  # 与真实标签比较
            total += data.y.size(0)  # 累加样本总数
    return correct / total  # 返回准确率


# --- 保存模型函数 ---
def save_model(model, epoch, save_dir):
    save_path = os.path.join(save_dir, f"gcn_model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存: {save_path}")


if __name__ == '__main__':
    # 初始化日志文件
    if os.path.exists(config['log_file']):
        os.remove(config['log_file'])  # 删除旧日志文件

    # --- 训练和测试 ---
    # 训练多个 epoch
    for epoch in range(config['epochs']):
        loss, accuracy = train(model, train_loader, optimizer, criterion, scheduler)
        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f},lr : {get_current_lr(optimizer)}")

        # 每5个epoch保存一次模型
        if (epoch + 1) % config['save_interval'] == 0:
            save_model(model, epoch + 1, config['save_dir'])

    input('训练完成，请按回车测试模型')
    # 测试模型
    accuracy = test(model, test_loader)
    print(f"Test Accuracy: {accuracy:.4f}")
