# -*- coding: utf-8 -*-
"""
Description: there is describe the script function
Environment: the script need the Environment
Example:     how to use the script
             $ python example_google.py
Author:      syf
Date:        2020.4.1
"""
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    """ load data from text .
    brief: each data must Split '   ' 3 space
    parameter:
        fileName: data.txt
    return:
        dataMat: matrix of data
    example: datMat = mat(loadDataSet('test.txt'))
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():                 # for each line
        curLine = line.strip().split('   ')
        fltLine = list(map(float, curLine))     # 这里和书中不同 和上一章一样修改
        dataMat.append(fltLine)
    return dataMat


#distance func
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB) 向量AB的欧式距离


def randCent(dataSet, k):
    """init K points randomly .
    brief: Randomly select the K centroid
    parameter:
        dataSet: after loadDataSet matrix
        k: multiple of centroid
    return:
        centroids: init centroids
    example:    datMat = mat(loadDataSet('test.txt'))
                centroids = randCent(datMat,2)
    """
    n = shape(dataSet)[1]                   # get matrix Column
    centroids = mat(zeros((k, n)))          # create centroid mat
    for j in range(n):                      # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])           # fine min in one column in data
        rangeJ = float(max(dataSet[:, j]) - minJ)    # calculate range
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """init K points randomly .
    brief: Randomly select the K centroid
    parameter:
        dataSet: after loadDataSet matrix
        k: multiple of centroid
    return:
        centroids: init centroids
    example:    datMat = mat(loadDataSet('test.txt'))
                centroids = randCent(datMat,2)
    """
    m, n = shape(dataMat) # 获取样本数和特征数
    clusterAssment = mat(zeros((m, 2)))     # 初始化一个矩阵来存储每个点的簇分配结果
                                            # clusterAssment包含两个列:一列记录簇索引值,
                                            # 第二列存储误差(误差是指当前点到簇质心的距离,
                                            # 后面会使用该误差来评价聚类的效果)
    centroids = createCent(dataMat, k)      # 创建质心,随机K个质心
    clusterChanged = True                   # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
    while clusterChanged:
        clusterChanged = False
        # 遍历所有数据找到距离每个点最近的质心,
        # 可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
        for i in range(m):
            minDist = inf  # 正无穷
            minIndex = -1
            for j in range(k):
                # 计算数据点到质心的距离
                # 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
                distJI = distMeas(centroids[j, :], dataMat[i, :])
                # 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 如果任一点的簇分配结果发生改变,则更新clusterChanged标志
            if clusterAssment[i, 0] != minIndex:
                # print(clusterAssment[i, 0],minIndex)
                clusterChanged = True
            # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)的平方
            clusterAssment[i, :] = minIndex, minDist ** 2
        # print(centroids)
        # 遍历所有质心并更新它们的取值
        for cent in range(k):
            # 通过数据过滤来获得给定簇的所有点
            ptsInClust = dataMat[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 计算所有点的均值,axis=0表示沿矩阵的列方向进行均值计算
            centroids[cent, :] = mean(ptsInClust, axis=0)  # axis=0列方向
    # 返回所有的类质心与点分配结果
    return centroids, clusterAssment


if __name__=="__main__":
    dataMat = mat(loadDataSet('test.txt'))
    centroids, clusterAssment = biKmeans(dataMat, 4)
    # for i in range(shape(datMat)[0]):
    #     x = datMat[i, 0]
    #     y = datMat[i, 1]
    #     plt.scatter(x,y, marker='x', color='red', s=40, label='First')
    # plt.show()
    print(centroids)
    print(clusterAssment)
    plt.scatter(array(dataMat)[:, 0], array(dataMat)[:, 1], c=array(clusterAssment)[:, 0].T)
    plt.scatter(centroids[:, 0].tolist(), centroids[:, 1].tolist(), marker='x')
    plt.show()