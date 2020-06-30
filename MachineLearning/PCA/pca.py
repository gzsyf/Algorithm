# -*- coding: utf-8 -*-
"""
Description: there is describe the script function
Environment: the script need the Environment
Example:     how to use the script
             $ python pca.py
Author:      syf
Date:        2020.5.3
"""

from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName, delim='\t'):
    """ load the data from text
    brief:
    parameter:
        fileName: the path of the file
        delim: segmentation string
    return:
        mat: a matrix
    example: dataMat = loadDataSet('testSet.txt')
    """
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat=999999):
    """ pca main
    brief:
    parameter:
        dataMat: data matrix
    return:
        lowDDataMat:
        reconMat:
    example: lowDMat, reconMat = pca(dataMat, 1)
    """
    meanVals = mean(dataMat, axis=0)                # Column mean (x and y mean)
    DataAdjust = dataMat - meanVals                 # 减去平均值
    covMat = cov(DataAdjust, rowvar=0)              # x y Covariance
    eigVals, eigVects = linalg.eig(mat(covMat))     # 计算特征值和特征向量
    # print eigVals
    eigValInd = argsort(eigVals)                    # Rank feature value
    print(eigVects)
    eigValInd = eigValInd[:-(topNfeat + 1):-1]      # 保留最大的前K个特征值
    redEigVects = eigVects[:, eigValInd]            # 对应的特征向量
    print(redEigVects)
    lowDDataMat = DataAdjust * redEigVects          # 将数据转换到低维新空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 升维，用于调试
    return lowDDataMat, reconMat

# Description: Virtual data
# Attention:
x= linspace(0,99,1000) - random.randint(-10, 10, (1,1000))
y= linspace(0,99,1000) - random.randint(-10, 10, (1,1000))
dataMat = vstack((x,y)).T

# Description: pca main
# Attention:
lowDMat, reconMat = pca(dataMat, 1)

# Description: draw picture
# Attention:
print(reconMat.shape)
plt.scatter(x,y)
plt.scatter(reconMat[:,0].flatten().A,reconMat[:,1].flatten().flatten().A)
plt.show()

