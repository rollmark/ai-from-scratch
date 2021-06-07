#!/usr/bin/env python
# -*-coding:utf-8-*-
# author:Mark.Qin
import torch


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