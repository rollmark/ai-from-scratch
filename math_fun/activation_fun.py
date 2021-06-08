#!/usr/bin/env python
# -*-coding:utf-8-*-
# author:Mark.Qin
import torch


### Rectified linear unit，ReLU: max(x,0)

def relu(x):
	relu =  torch.nn.ReLU()
	return relu(x)
	#a = torch.zeros_like(x)
	#return torch.max(x, a)


### sigmoid函数将输入变换为区间(0, 1)上的输出
def sigmoid(x):
	return 1/(1+torch.exp(-x))


### tanh(双曲正切)函数也能将其输入压缩转换到区间(-1, 1)上
def tanh(x):
	return (1-torch.exp(-2*x)) / (1+ torch.exp(-2*x))