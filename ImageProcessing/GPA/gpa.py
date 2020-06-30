# -*- coding: utf-8 -*-
"""
Description: GPA General Procrustes Analysis
Environment: the script need the Environment
Example:     how to use the script
             $ python example_google.py
Author:      syf
Date:        2020.4.1
"""

import numpy as np
import face_recognition
import cv2
from PIL import Image,ImageDraw
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


def transformation_from_points(points1, points2):
    """ main function of gpa.
    brief:
        xxx
    parameter:
        points1: the image need to rotate feature (n*2 matrix)
        points2: target picture features (n*2 matrix)
    return:
        trans_mat: rotate matrix
    example:
        R = transformation_from_points(np.array(D).T,np.array(B).T)
    """

    '''0 - 先确定是float数据类型 '''
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    '''1 - 消除平移的影响 '''
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    '''2 - 消除缩放的影响 '''
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    '''3 - 计算矩阵M=BA^T；对矩阵M进行SVD分解；计算得到R '''
    # ||RA-B||; M=BA^T
    A = points1.T # 2xN
    B = points2.T # 2xN
    M = np.dot(B, A.T)
    U, S, Vt = np.linalg.svd(M)
    R = np.dot(U, Vt)

    '''4 - 构建仿射变换矩阵 '''
    s = s2/s1
    sR = s*R
    c1 = c1.reshape(2,1)
    c2 = c2.reshape(2,1)
    T = c2 - np.dot(sR,c1) # 模板人脸的中心位置减去 需要对齐的中心位置（经过旋转和缩放之后）

    trans_mat = np.hstack([sR,T])   # 2x3

    return trans_mat


def warp_im(in_image, trans_mat, dst_size):
    """ 用旋转矩阵去旋转目标图片 .
    brief:
        xxx
    parameter:
        in_image: 目标图片
        trans_mat: 旋转矩阵
        dst_size: zoom size
    return:
        output_image:
    example:
        img = warp_im(image_array2, R, (image_array1.shape[1], image_array1.shape[0]))
    """
    output_image = cv2.warpAffine(in_image,
                                  trans_mat,
                                  dst_size,  # (cols, rows)
                                  borderMode=cv2.BORDER_TRANSPARENT)
    return output_image


''' 读取图片 '''
image_array1 = cv2.imread(r'C:\Users\gzsyf\Pictures\Saved Pictures\girl2.png')
image_array2 = cv2.imread(r'C:\Users\gzsyf\Pictures\Saved Pictures\people.jpg')

''' 提取面部特征点 '''
face_landmarks_list1 = face_recognition.face_landmarks(image_array1, model="large")
face_landmarks_dict1 = face_landmarks_list1[0]
face_landmarks_list2 = face_recognition.face_landmarks(image_array2, model="large")
face_landmarks_dict2 = face_landmarks_list2[0]

''' 对特征矩阵变形，变成n*2的结构 '''
print('***************************************************')
A=[]
B=[[],[]]
for k in face_landmarks_dict1:
    A.extend(face_landmarks_dict1[k])
for i in A:
    B[0].append(i[0])
    B[1].append(i[1])
C=[]
D=[[],[]]
for k in face_landmarks_dict2:
    C.extend(face_landmarks_dict2[k])
for i in C:
    D[0].append(i[0])
    D[1].append(i[1])


''' gpa得到旋转矩阵 '''
R = transformation_from_points(np.array(D).T,np.array(B).T)

''' 展示图片 '''
plt.subplot(121)
imshow(Image.fromarray(image_array2))
plt.subplot(122)
img = warp_im(image_array2, R, (image_array1.shape[1], image_array1.shape[0]))
imshow(Image.fromarray(img))
plt.show()

"""Reference
[1] code come from & theory
https://www.cnblogs.com/shouhuxianjian/p/10058174.html
[2] other
https://blog.csdn.net/lien0906/article/details/52208888
"""