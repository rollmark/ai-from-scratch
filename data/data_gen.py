#!/usr/bin/env python
# -*-coding:utf-8-*-
# author:Mark.Qin
import random

import math
import torch
from sklearn.datasets import make_moons
import numpy as np

### Generate linera data
def synthetic_data(w,b,num_examples):
	"""生成 y = Xw + b + 噪声。"""
	X = torch.normal(0,1,size=(num_examples,len(w)))
	y = torch.matmul(X,w) + b
	y += torch.normal(0,0.001, size=y.shape)
	return X,y.reshape(-1,1)


### Generate binary clf data
def generate_binary_clf_data(w,b,num_examples):
	X,y = synthetic_data(w,b,num_examples)
	y = torch.sigmoid(y)
	return X,y

### generate binary dta
def generate_binary_data():
	X,y = make_moons(500,0.2)
	return X,y


### generate poolynomial data
def generate_polynomial():
	max_degree = 20  # 多项式的最大阶数
	n_train, n_test = 100, 100  # 训练和测试数据集大小
	true_w = np.zeros(max_degree)  # 分配大量的空间
	true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

	features = np.random.normal(size=(n_train + n_test, 1))
	np.random.shuffle(features)
	poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
	for i in range(max_degree):
		poly_features[:, i] /= math.gamma(i + 1)  # `gamma(n)` = (n-1)!
	# `labels`的维度: (`n_train` + `n_test`,)
	labels = np.dot(poly_features, true_w)
	labels += np.random.normal(scale=0.1, size=labels.shape)

