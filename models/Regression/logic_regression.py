#!/usr/bin/env python
# -*-coding:utf-8-*-
# author:Mark.Qin
from sklearn.model_selection import train_test_split

from data.data_gen import synthetic_data, generate_binary_data, generate_binary_clf_data
from data.batch_gen import data_iter
import torch
from math_fun.model_fun import binary_logic_reg
from math_fun.loss_fun import binary_cross_entroy_loss
from optimization.optimization import sgd


class BinaryLogicRegression:
	def __init__(self, w, b,train_x,train_y,learning_rate,batch_size,num_epochs):
		self.w = w
		self.b = b
		self.train_x = train_x
		self.train_y = train_y
		self.net = binary_logic_reg
		self.loss = binary_cross_entroy_loss
		#self.loss = torch.nn.BCELoss()
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
				# print("W:{}".format(self.w))
				# print("B:{}".format(self.b))
				# print("--------------------")
			with torch.no_grad():
				y_pred = self.forward(self.train_x)
				train_l = self.loss(y_pred, self.train_y)
				print('epoch:{}, loss:{}'.format(epoch + 1, float(train_l.mean())))

	def inference(self,x):
		return self.forward(x)


X,y = generate_binary_data()
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.25,random_state=73)

## sklearn version
''' 
from sklearn.linear_model import LogisticRegression as lrg

clf = lrg(random_state=0).fit(X,y)
print(clf.coef_) # [[ 1.09497619 -4.85816999]]
print(clf.intercept_) # [0.66705483]
print(clf.score(X,y)) # 0.884
'''


#torch version with sgd

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).reshape(-1,1)


#训练有关参数
learning_rate = 0.03
num_epochs = 1000
batch_size = 20

#初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 设置参数W和b
true_w = torch.tensor(([2,-3.4]))
true_b = 4.2

X_train_t,y_train_t = generate_binary_clf_data(true_w,true_b,1000)


blr = BinaryLogicRegression(w,b,X_train_t,y_train_t,learning_rate,batch_size,num_epochs)
blr.train()

print(blr.w)
print(blr.b)

# print(blr.forward(X_train_t[0]))
# print(y_train_t[0])


