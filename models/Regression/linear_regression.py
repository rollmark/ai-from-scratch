#!/usr/bin/env python
# -*-coding:utf-8-*-
# author:Mark.Qin

from data.data_gen import synthetic_data
from data.batch_gen import data_iter
import torch
from math_fun.model_fun import linreg
from math_fun.loss_fun import squared_loss
from optimization.optimization import sgd


class LinearRegression:
	def __init__(self, w, b,train_x,train_y,learning_rate,batch_size,num_epochs):
		self.w = w
		self.b = b
		self.train_x = train_x
		self.train_y = train_y
		self.net = linreg
		self.loss = squared_loss
		self.optimization = sgd
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.num_epochs = num_epochs

	def forward(self,X):
		return self.net(X,self.w, self.b)

	def train(self):
		# 训练
		for epoch in range(self.num_epochs):
			for X, y in data_iter(self.batch_size, self.train_x, self.train_y):
				y_hat = self.forward(X)
				l = self.loss(y_hat, y)  # `X`和`y`的小批量损失
				# 因为`l`形状是(`batch_size`, 1)，而不是一个标量。`l`中的所有元素被加到一起，
				# 并以此计算关于[`w`, `b`]的梯度
				l.sum().backward()
				sgd([self.w, self.b], self.learning_rate, self.batch_size)  # 使用参数的梯度更新参数
			with torch.no_grad():
				y_pred = self.forward(self.train_x)
				train_l = self.loss(y_pred, self.train_y)
				print('epoch:{}, loss:{}'.format(epoch + 1, float(train_l.mean())))

	def inference(self,x):
		return self.forward(x)



#训练有关参数
learning_rate = 0.03
num_epochs = 3
batch_size = 10

#初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 设置参数W和b
true_w = torch.tensor(([2,-3.4]))
true_b = 4.2

# 伪造数据
train_x,train_y = synthetic_data(true_w,true_b,1000)

lr = LinearRegression(w,b,train_x,train_y,learning_rate,batch_size,num_epochs)
lr.train()

print(lr.w)
print(lr.b)

test_x, test_y = synthetic_data(true_w,true_b,2)
pred_y = lr.inference(test_x)

print(test_y)
print(pred_y)



