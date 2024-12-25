"""
(utils)
这个脚本用来生成jpg->3d坐标点阵图
"""
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 初始化 Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# 读取 JPG 图片
image_path = "./datasets/images/1225.jpg"  # 替换为你的图片路径
image = cv2.imread(image_path)

if image is None:
    print("无法加载图像，请检查路径。")
    exit()

# 转换为 RGB 格式（Mediapipe 需要）
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 处理图像以提取人脸关键点
results = face_mesh.process(rgb_image)

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # 提取关键点坐标
        h, w, _ = rgb_image.shape
        keypoints_x = [landmark.x for landmark in face_landmarks.landmark]
        keypoints_y = [landmark.y for landmark in face_landmarks.landmark]
        keypoints_z = [landmark.z for landmark in face_landmarks.landmark]  # 相对深度值

        # 创建 3D 散点图
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制关键点
        ax.scatter(keypoints_x, keypoints_y, keypoints_z, c='blue', s=10)

        # 设置轴标签
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Face Keypoints Visualization")

        # 调整视角
        ax.view_init(elev=10, azim=60)

        plt.show()

else:
    print("未检测到人脸。")
