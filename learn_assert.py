import numpy as np
import cv2

# 定义3D点
points = np.array([[[0, 0, 0]], [[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]], dtype=np.float32)
# 定义相机内参矩阵K和畸变系数distCoeffs（这里仅作示例）
K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
distCoeffs = np.zeros((5, 1), dtype=np.float32)  # 无畸变假设
# 定义旋转矩阵和平移向量（这里仅作示例）
rotationMatrix = np.eye(3, dtype=np.float32)  # 无旋转假设
translationVector = np.zeros((3, 1), dtype=np.float32)  # 无平移假设
# 调用cv2.projectPoints函数进行投影
dstPoints = cv2.projectPoints(points, rotationMatrix, translationVector, K, distCoeffs)