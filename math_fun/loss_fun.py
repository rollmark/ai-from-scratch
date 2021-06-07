#!/usr/bin/env python
# -*-coding:utf-8-*-
# author:Mark.Qin
import torch

def squared_loss(y_hat,y):
	return (y_hat - y.reshape(y_hat.shape)) **2 / 2


def binary_cross_entroy_loss(y_hat, y):
	return -y*torch.log(y_hat) - (1-y)*torch.log(1-y_hat)


def softmax_cross_entropy_reg_loss(y_hat,y):
	#y:one-hotel labels
	# loss = -y_i * log(y_hat_i)
	# y_i:[0,0,...1,0]
	return -torch.log(y_hat[range(len(y_hat)), y])

