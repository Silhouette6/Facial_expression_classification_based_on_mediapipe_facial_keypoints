"""
(utils)
这个脚本用来将批量构建4x478的npy数组
需要读取带名字索引的3x478npy文件和打好的csv标签文件
"""

import csv
import numpy as np
import os

if __name__ == "__main__":
    # 定义输入输出路径
    csv_file = "./datasets/labels/1-1600.csv"
    output_dir = "./labels_output"
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)

        # 计算总行数并重置文件指针
        total_rows = sum(1 for _ in reader)
        file.seek(0)  # 重置文件指针
        print(f"CSV 总行数: {total_rows}")

        row_count = 0
        for row in reader:
            # 跳过无效行
            if len(row) < 2:
                print(f"跳过无效行: {row}")
                continue

            row_count += 1
            index, label = row[0], row[1]

            npy_file_path = os.path.join("./output", f"{index}.npy")
            npy_file_path_updated = os.path.join(output_dir, f"{index}.npy")

            if not os.path.exists(npy_file_path):
                print(f"文件 {npy_file_path} 不存在！")
            else:
                # 加载 .npy 文件
                data = np.load(npy_file_path)  # 原始数据为 (3, 478)

                # 创建要添加的行
                try:
                    label = float(label)  # 转换标签为数值
                    additional_row = np.full((1, data.shape[1]), label)  # 形状为 (1, 478)

                    # 拼接到第 4 行
                    updated_data = np.vstack((data, additional_row))  # 拼接后为 (4, 478)

                    # 保存更新后的 .npy 文件
                    np.save(npy_file_path_updated, updated_data)
                except ValueError:
                    print(f"标签转换失败: {label}, 跳过此行")

            print(f"{row_count}/{total_rows}")





