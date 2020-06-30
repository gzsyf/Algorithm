# -*- coding: utf-8 -*-
"""
Description: Hog
Environment: the script need the Environment
Example:     how to use the script
             $ python example_google.py
Author:      syf
Date:        2020.4.1
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


class Hog_descriptor():
    """ Hog Direction gradient histogram.
    brief:
        xxx
    member:
        img: Original picture
        cell_size: To see theoretical knowledge
        bin_size: 360 degree resolution
    method:
        extract():
        global_gradient():
        cell_gradient(cell_magnitude, cell_angle):
        get_closest_bins(gradient_angle):
        render_gradient(image, cell_gradient):
    example:
        hog = Hog_descriptor(img, cell_size=8, bin_size=8)
    """

    def __init__(self, img, cell_size=16, bin_size=8):
        self.img = img                              # 图片
        self.img = np.sqrt(img / np.max(img))       # 图片色彩归一化
        self.img = img * 255                        # ？？？
        self.cell_size = cell_size                  # cell的大小
        self.bin_size = bin_size                    # 向量的分辨率
        self.angle_unit = 360 // self.bin_size      # 360度分开
        assert type(self.bin_size) == int,    "bin_size should be integer,"
        assert type(self.cell_size) == int,   "cell_size should be integer,"
        assert type(self.angle_unit) == int,  "bin_size should be divisible by 360"

    def extract(self):
        """ Extract feature matrix .
        brief:
            the main method in this class
        parameter:
            None
        return:
            hog_vector： feature matrix
            hog_image： Visualizing gradient image
        example:
            vector, image = hog.extract()
        """
        # Description: 求每个像素的梯度(大小和方向)
        # Attention:
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)

        # Description: 分成网格，再进行直方图提取信息
        # Attention:
        cell_gradient_vector = np.zeros((height // self.cell_size, width // self.cell_size, self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                # 用直方图 给出每个cell的方向
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        # Description: 可视化梯度图像
        # Attention:
        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)

        # Description: 统计Block的梯度信息
        # Attention:
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element // magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector, hog_image


    def global_gradient(self):
        """ Find the gradient of each element .
        brief:
            xxxx
        parameter:
            None
        return:
            gradient_magnitude:
            gradient_angle:
        example:
            xxx
        """
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)  # x轴的梯度算子卷积核
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)  # y轴的梯度算子卷积核
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)  # 梯度的大小
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)  # 梯度的方向
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        """Histogram of gradient for each cell .
        brief:
            xxxx
        parameter:
            cell_magnitude: Elements of the entire cell
            cell_angle: Elements of the entire cell
        return:
            orientation_centers: the main gradient
        example:
            xxx
        """
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                # Description: 分配权值
                # Attention: 用网页里面的第二种方法
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        """ xxxx .
        brief:
            xxxx
        parameter:
            gradient_angle:
        return:
            min_angle:
            max_angle:
            mod:
        example:
            xxx
        """
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        """ Visualize Cell Gradient Histogram .
        brief:
            xxxx
        parameter:
            image: empty image with image shape
            cell_gradient: The cell gradient after the histogram
        return:
            image: gradient image
        example:
            xxx
        """
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

# Description: 读取图片
# Attention:
img = cv2.imread(r'C:\Users\gzsyf\Pictures\Saved Pictures\equelizehist.jpg', cv2.IMREAD_GRAYSCALE)
# Description: 提取特征
# Attention:
hog = Hog_descriptor(img, cell_size=8, bin_size=8)
vector, image = hog.extract()
print (np.array(vector).shape)
plt.imshow(image, cmap=plt.cm.gray)
plt.show()

"""Reference
[1] code come from
https://blog.csdn.net/ppp8300885/article/details/71078555
https://blog.csdn.net/wsp_1138886114/article/details/82964639
[2] code theoretical knowledge
https://blog.csdn.net/u013066730/article/details/83015490#%E7%89%B9%E5%BE%81%E6%8F%8F%E8%BF%B0%E5%AD%90(Feature%20Descriptor)
"""