#!/usr/bin/env python
# -*-coding:utf-8-*-
# author:Mark.Qin

## 构建batch
import random

import torch


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    #print(indices)
    for i in range(0, num_examples, batch_size):
        #print(i)
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        #print(batch_indices)
        yield features[batch_indices], labels[batch_indices]