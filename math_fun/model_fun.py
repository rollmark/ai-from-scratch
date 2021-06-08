#!/usr/bin/env python
# -*-coding:utf-8-*-
# author:Mark.Qin
import torch

from math_fun.activation_fun import relu


def linreg(X,w,b):
	return torch.matmul(X,w) + b

def binary_logic_reg(X,w,b):
	return 1/ (1+torch.exp(-linreg(X,w,b)))

def softmax(X):
	X_exp = torch.exp(X)
	partition = X_exp.sum(1, keepdim=True)
	return X_exp / partition

def soft_reg_net(X,W,b):
	return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def mlp(X,w1,b1,w2,b2):
	#flatten X
	X= X.reshape(-1,784)
	#Z = X @ w1 + b1
	Z = torch.matmul(X,w1) + b1
	H = relu(Z)
	#out_put = H @ w2 + b2
	out_put = torch.matmul(H,w2) + b2
	return out_put
