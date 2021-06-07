#!/usr/bin/env python
# -*-coding:utf-8-*-
# author:Mark.Qin

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
# 并除以255使得所有像素的数值均在0到1之间
import torchvision
from torchvision import transforms
from torch.utils import data

data_root_path = "/Users/mark.qin/Mark/mark_github/ai-from-scratch/data/fashion_mnist"

def download_data():
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root=data_root_path, train=True,
                                                    transform=trans,
                                                    download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root=data_root_path, train=False,
                                                   transform=trans, download=False)

    return mnist_train,mnist_test


#print(mnist_train.shape)
# print(mnist_test[0][0][0])


batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据。"""
    return 1

def get_batch_iter(batch_size):
    mnist_train, mnist_test = download_data()
    print(len(mnist_train), len(mnist_test))
    return (data.DataLoader(mnist_train, batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False)
            )


#train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             #num_workers=get_dataloader_workers())




