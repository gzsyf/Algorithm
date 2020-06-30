# -*- coding: utf-8 -*-
"""
Description: lda algorithm
Environment: the script need the Environment
Example:     how to use the script
             $ python example_google.py
Author:      syf
Date:        2020.5.7
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def lda(data, target, n_dim):
    """ lda algorithm.
    brief: calculate most important eigVects
    parameter:
        data:       n_samples, n_features
        target:     data class
        n_dim:      want to decrease dimension
    return: n_samples, n_dims
    example:
    """

    clusters = np.unique(target)

    # Description: error
    # Attention: 要降维嘛 设置维度不能高于原来的
    if n_dim > len(clusters)-1:
        print("K is too much")
        print("please input again")
        exit(0)

    # Description: calculate within_class scatter matrix
    # Attention:
    Sw = np.zeros((data.shape[1],data.shape[1]))
    for i in clusters:
        datai = data[target == i]
        datai = datai-datai.mean(0)
        Swi = np.mat(datai).T*np.mat(datai)
        Sw += Swi

    # Description: calculate between_class scatter matrix
    # Attention:
    SB = np.zeros((data.shape[1],data.shape[1]))
    u = data.mean(0)  #所有样本的平均值
    for i in clusters:
        Ni = data[target == i].shape[0]
        ui = data[target == i].mean(0)  #某个类别的平均值
        SBi = Ni*np.mat(ui - u).T*np.mat(ui - u)
        SB += SBi

    # Description: calculate eigVects
    # Attention:
    S = np.linalg.inv(Sw)*SB
    eigVals,eigVects = np.linalg.eig(S)  #求特征值，特征向量
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:(-n_dim-1):-1]
    w = eigVects[:,eigValInd]
    data_ndim = np.dot(data, w)

    return data_ndim

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target

    data_1 = lda(X, Y, 2)

    plt.title("LDA")
    plt.scatter(data_1[:, 0], data_1[:, 1], c = Y)
    plt.show()