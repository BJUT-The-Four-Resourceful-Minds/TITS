import torch
import numpy as np


def likelihood_transformation(X):
    M, N = X.shape#得到输入数据矩阵的形状（即行数和列数）
    mu_m = np.mean(X, axis=1)#求特征均值
    sigma_m = np.var(X, axis=1)#求特征方差
    Y = np.zeros((M, N))#构建元素全为零的同形状的矩阵用于存储每个元素的特征似然值

    for m in range(M):#通过内外两层循环遍历矩阵中的每一个元素
        for n in range(N):
            Y[m, n] = (1 / np.sqrt(2 * np.pi * sigma_m[m])) * np.exp(
                -((X[m, n] - mu_m[m]) ** 2) / (2 * sigma_m[m])
            )#该式为正态分布的概率密度函数 用于求出每一个元素的特征似然值

    z = np.sum(Y, axis=0) / np.max(np.sum(Y, axis=0))#实现归一化 使得所有聚合似然值在经过归一化后范围在0-1之间
    return z

#以下为测试代码

example_data = np.random.rand(5, 10)

result = likelihood_transformation(example_data)
print("归一化似然变换结果:")
print(result)