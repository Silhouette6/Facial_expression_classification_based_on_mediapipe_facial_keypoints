"""
(utils)
这个脚本用来show一个csv...
"""
import pandas as pd

# 加载 CSV 文件
csv_file = "./datasets/labels/1-1600.csv"
df = pd.read_csv(csv_file)

# 查看前 5 行数据
print(df.head())

# 查看数据的形状 (行数, 列数)
print("数据形状:", df.shape)

# 查看列名
print("列名:", df.columns)

# 查看统计信息
print(df.describe())

input('随意按键继续')

# 打开 CSV 文件
import csv

# 打开 CSV 文件
with open(csv_file, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)

    # 打印每一行
    for row in reader:
        print(row)
