# GCN-Based Facial Expression Classification

## Overview

This project implements a Graph Convolutional Network (GCN) model to classify facial expressions into two categories: **smiling** or **not smiling**. The workflow includes processing binary image files, converting them to JPEG format, generating graphs with MediaPipe, and training a GCN for classification.

## Workflow

1. **Binary Image Processing**: Convert raw binary image files into JPEG format for visualization and further processing.
2. **Graph Generation**: Utilize [MediaPipe](https://mediapipe.dev/) to extract keypoints from facial images and generate graph representations.
3. **GCN Model Training**: Train a Graph Convolutional Network using the generated graphs to classify facial expressions.


## Prerequisites

### Dependencies

- Python 3.8+

- Required Python libraries:

  ```bash
  torch torchvision torch-geometric mediapipe matplotlib pickle...
  ```

### Dataset

- Binary image files of faces.
- Prepare train and test datasets as follows:
  - Convert binary images to JPEG.
  - Use MediaPipe to extract graph structures.
  - Save graph data as `.pkl` files in `./datasets/graphs/train` and `./datasets/graphs/test` directories.

## Project Structure

```
project_root/
|— datasets/
|    |— graphs/
|         |— train/       # Training graph data (.pkl files)
|         |— test/        # Testing graph data (.pkl files)
|— saved_models/         # Directory to save trained models
|— log/	      # Training logs
|— train.py          # train the GCN model
|- test.py			 # test the GCN model
|— utils/	             # Utility functions (e.g., data processing, logging, visualization)
|— README.md             # Project documentation
```

## Usage

### 1. Prepare Data（see Utils）

1. Convert binary images to JPEG format.
2. Use MediaPipe to generate graphs and save them as `.pkl` files.

### 2. Train the Model

Run the training script:

```bash
python train.py
```

## Configuration

Hyperparameters can be adjusted in the `config` dictionary in `gcn_model.py`. Example:

```python
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
```

## 

Training logs will be saved in `training_log.txt`, and models will be periodically saved in the `saved_models/` directory.

### 3. Visualize Training Metrics

Use the provided visualization script to plot loss, accuracy, and learning rate:

```bash
python ./Utils/Visual_training.py
```

### 4. Test the Model

Evaluate the model on the test dataset:

```bash
python test.py
```

## Configuration

Hyperparameters can be adjusted in the `config` dictionary in `gcn_model.py`. Example:

```python
config = {
    'input_dim': 3,  # 输入特征维度（3D 坐标）
    'hidden_dim': 256,  # 隐藏层的维度
    'output_dim': 2,  # 输出维度（2：微笑与否）
    'batch_size': 32,  # 每个批次的数据量
    'save_dir': './saved_models',  # 模型保存路径文件夹
    'model_name': 'best1.pth'  # 模型文件名
}
```

## Results

- The model achieves approximately **83.3% accuracy** on the test set.
- Training metrics such as loss and accuracy can be visualized for deeper insights.

## Acknowledgments

This project uses the following libraries and tools:

- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [MediaPipe](https://mediapipe.dev/)
- [Matplotlib](https://matplotlib.org/)
