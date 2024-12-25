"""
(utils)
这个脚本用来批量将jpg文件构建成3x478的npy文件
"""
import os
import cv2
import numpy as np
import mediapipe as mp

# 配置输入和输出路径
input_folder = "./datasets/images"  # 替换为你的图像目录路径
output_folder = "./output"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 初始化 Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# 遍历输入文件夹下的所有 .jpg 图片
for file_name in os.listdir(input_folder):
    if file_name.endswith(".jpg"):
        image_path = os.path.join(input_folder, file_name)

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法加载图像：{file_name}")
            continue

        # 转换为 RGB 格式
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 提取人脸关键点
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            print(f"未检测到人脸：{file_name}")
            continue

        # 获取关键点的 3D 坐标 (x, y, z)
        for face_landmarks in results.multi_face_landmarks:
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]).T  # 转置为 3x478

            # 构造输出文件路径
            output_file_name = os.path.splitext(file_name)[0] + ".npy"
            output_file_path = os.path.join(output_folder, output_file_name)

            # 保存为 .npy 文件
            np.save(output_file_path, keypoints)
            print(f"保存成功：{output_file_path}")
