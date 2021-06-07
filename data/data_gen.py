#!/usr/bin/env python
# -*-coding:utf-8-*-
# author:Mark.Qin
import random
import torch
from sklearn.datasets import make_moons

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