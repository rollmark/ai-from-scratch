#!/usr/bin/env python
# -*-coding:utf-8-*-
# author:Mark.Qin
import torch

from data.fashion_mnist_data import get_batch_iter
from math_fun.loss_fun import softmax_cross_entropy_reg_loss
from math_fun.model_fun import soft_reg_net
from math_fun.performance_fun import Accumulator, accuracy, evaluate_accuracy
from optimization.optimization import sgd


class SoftMaxRegression:
	def __init__(self,w,b,num_inputs, num_outputs, learning_rate, batch_size, num_epochs):
		self.w = w
		self.b = b
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.net = soft_reg_net
		self.loss = softmax_cross_entropy_reg_loss
		self.optimization = sgd

	def forward(self,x):
		return self.net(x,self.w,self.b)

	def train(self):
		train_iter,test_iter = get_batch_iter(self.batch_size)
		metric = Accumulator(3)
		for epoch in range(self.num_epochs):
			for X, y in train_iter:
				y_hat = self.forward(X)
				l = self.loss(y_hat, y)
				l.sum().backward()
				sgd([self.w, self.b], self.learning_rate, self.batch_size)  # 使用参数的梯度更新参数
				metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
			with torch.no_grad():
				res = Accumulator(3)  # 正确预测数、预测总数

				for X, y in test_iter:
					y_hat = self.forward(X)
					l = self.loss(y_hat,y)
					res.add(float(l.sum()),accuracy(y_hat, y), y.numel())
				acc = res[1] / res[2]
				loss = res[0] / res[2]
				print("Epoch:{}, test accu: {}, test loss: {}".format(epoch, acc,loss))



## 初始化参数
num_inputs = 784  # 28*28 打平
num_outputs = 10
batch_size = 256
learning_rate = 0.1
num_epochs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

if __name__ == '__main__':
	smr = SoftMaxRegression(W, b, num_inputs, num_outputs, learning_rate, batch_size, num_epochs)
	smr.train()
	print(smr.w)
	print(smr.b)