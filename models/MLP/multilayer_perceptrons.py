#!/usr/bin/env python
# -*-coding:utf-8-*-
# author:Mark.Qin

## 多层感知机/全联接网络
from torch import nn

from data.fashion_mnist_data import get_batch_iter
from math_fun.activation_fun import relu
from math_fun.model_fun import mlp
from math_fun.performance_fun import Accumulator, accuracy
from optimization.optimization import sgd
import torch


class MLP:
	def __init__(self, w1,b1, w2,b2,num_inputs,num_hiddens,num_outputs, learning_rate, batch_size,num_epochs):
		self.w1 = w1
		self.b1 = b1
		self.w2 = w2
		self.b2 = b2
		self.num_inputs = num_inputs
		self.num_hiddens = num_hiddens
		self.num_outputs = num_outputs
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.net = mlp
		self.act = relu
		self.optimization = sgd
		self.loss = torch.nn.CrossEntropyLoss()

	def forward(self,X):
		return self.net(X,self.w1,self.b1,self.w2,self.b2)

	def train(self):
		train_iter, test_iter = get_batch_iter(self.batch_size)
		for epoch in range(self.num_epochs):
			res = Accumulator(3)  # 正确预测数、预测总数
			for X,y in train_iter:
				y_hat = self.forward(X)
				l = self.loss(y_hat, y)
				l.backward()
				sgd(params,self.learning_rate, self.batch_size)
			with torch.no_grad():

				for x,y in test_iter:
					y_pred = self.forward(x)
					l = self.loss(y_pred, y)
					res.add(float(l), accuracy(y_pred, y), y.numel())
			acc = res[1] / res[2]
			loss = res[0] / res[2]
			print("Epoch:{}, test accu: {}, test loss: {}".format(epoch, acc, loss))


## init parameters
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(
    torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(
    torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]


## 初始化参数
num_inputs = 784  # 28*28 打平
num_hiddens = 256
num_outputs = 10

batch_size = 256
learning_rate = 10
num_epochs = 20


if __name__ == '__main__':
	m = MLP(W1,b1,W2,b2, num_inputs,num_hiddens, num_outputs, learning_rate, batch_size, num_epochs)
	m.train()
	# print(m.w1)
	# print(m.b1)
	# print(m.w2)
	# print(m.b2)
