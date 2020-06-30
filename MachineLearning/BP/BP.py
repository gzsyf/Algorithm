# -*- coding: utf-8 -*-
"""
Description: there is describe the script function
Environment: the script need the Environment
Example:     how to use the script
             $ python example_google.py
Author:      syf
Date:        2020.4.1
"""

import math
import random

random.seed(0)



def rand(a, b):
    """ 重定义随机函数.
    brief:
        xxx
    parameter:
        a,b:产生的随机数在ab之间
    return:
        xxx: 随机数
    example:
        xxxx
    """
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    """ 创造一个指定大小的矩阵 .
    brief:
        xxx
    parameter:
        m, n: m*n的矩阵
        fill:要填某个数
    return:
        mat: 矩阵
    example:
        xxxx
    """
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    """ 定义sigmod函数 .
    brief:
        xxx
    parameter:
        xxx:
    return:
        xxx:
    example:
        xxxx
    """
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    """ 定义sigmod函数的导数 .
    brief:
        xxx
    parameter:
        xxx:
    return:
        xxx:
    example:
        xxxx
    """
    return x * (1 - x)


class BPNeuralNetwork:
    """one sentence describe .
    brief:
        xxx
    member:
        xxx: 很多在 init有注释
    method:
        setup(self, ni, nh, no): 初始化
        predict(self, inputs): 前向传导
        back_propagate(self, case, label, learn, correct): 反向传导
        train(self, cases, labels, limit=10000, learn=0.05, correct=0.1): 训练
        test(self): 测试
    example:
        xxxx
    """
    def __init__(self):
        self.input_n = 0                # 输入层神经元的个数
        self.hidden_n = 0               # 隐藏层神经元的个数
        self.output_n = 0               # 输出层神经元的个数

        self.input_cells = []           # 输入层神经元的值
        self.hidden_cells = []          # 隐藏层神经元的值
        self.output_cells = []          # 输出层神经元的值

        self.input_weights = []         # 输入层权重
        self.output_weights = []        # 输入层权重

        self.input_correction = []      # 矫正矩阵
        self.output_correction = []     # 矫正矩阵

    def setup(self, ni, nh, no):
        """ 初始化权重偏置等 .
        brief:
            xxxx
        parameter:
            ni:输入层神经元的个数
            nh:隐藏层神经元的个数
            no:输出层神经元的个数
        return:
            None:
        example:
            xxx
        """
        self.input_n = ni + 1           # 输入层神经元的个数
        self.hidden_n = nh              # 隐藏层神经元的个数
        self.output_n = no              # 输出层神经元的个数

        ''' init cells 初始化神经元的初值 '''
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n

        ''' init weights 创建权重数组 '''
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)

        ''' random activate  初始化两个权重 (随机法) '''
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)

        ''' init correction matrix 反向传导矩阵 '''
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    def predict(self, inputs):
        """ 正向传导 .
        brief:
            xxxx
        parameter:
            inputs: 训练数据
        return:
            output_cells: 输出层
        example:
            xxx
        """
        '''activate input layer 赋予输入层每个神经元初值'''
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]

        '''activate hidden layer  传到隐藏层'''
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]     # 隐藏层权重
            self.hidden_cells[j] = sigmoid(total)                           # 激活

        '''activate output layer 传到输出层'''
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]   # 输出层权重
            self.output_cells[k] = sigmoid(total)                           # 激活

        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        """ 反向传播 .
        brief:
            xxx:
        parameter:
            case: 数据
            label: 标签
            learn: 学习率
            correct: 矫正率
        return:
            error: 总误差
        example:
            xxx
        """
        '''feed forward 先正向传导'''
        self.predict(case)

        '''get output layer error 求出输出层的偏差'''
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error

        '''get hidden layer error 求隐藏层的偏差'''
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error # 有点出入

        '''update output weights 更新输出层的偏差'''
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change

        '''update input weights 更新输出层的偏差'''
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change

        '''get global error 损失函数'''
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    # 训练 limit最大迭代次数， learn学习率， correct矫正率
    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        # 开始迭代
        for j in range(limit):
            error = 0.0

            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)

    def test(self):
        cases = [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
        ]
        labels = [[0], [0], [0], [0]]
        self.setup(3, 3, 1)
        self.train(cases, labels, 30000, 0.05, 0.1)

        for case in cases:
            print(self.predict(case))


if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.test()
