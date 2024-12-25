import matplotlib.pyplot as plt


# 读取训练日志文件并解析其中的 Loss 和 Accuracy
def read_training_log(log_file):
    epochs = []
    losses = []
    accuracies = []

    with open(log_file, 'r') as f:
        for line in f:
            # 每一行都应该包含类似于 "Epoch X: Loss: Y, Accuracy: Z"
            if 'Epoch' in line:
                parts = line.strip().split(', ')

                epoch = int(parts[0].split(' ')[1].replace(',Loss:',''))  # 提取 Epoch 数字
                loss = float(parts[0].split(' ')[2].replace(',Accuracy:',''))  # 提取 Loss 数字
                accuracy = float(parts[0].split(' ')[3].replace(',lr:',''))  # 提取 Accuracy 数字
                print(epoch, loss, accuracy)

                epochs.append(epoch)
                losses.append(loss)
                accuracies.append(accuracy)

    return epochs, losses, accuracies

if __name__ == '__main__':
    # 从日志文件中读取数据
    log_file = '../log/training_log_12241727.txt'

    epochs, losses, accuracies = read_training_log(log_file)

    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 6))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label="Loss", color='r', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label="Accuracy", color='g', marker='o')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # 显示图形
    plt.tight_layout()
    plt.show()
