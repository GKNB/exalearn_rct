#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch import FloatTensor
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler
import horovod.torch as hvd 
import io, os, sys
import time

data = np.ones((100000,3))
res = np.ones(100000)

data1_x = torch.from_numpy(data) * 2.0
print(data1_x.shape)
print(data1_x.dtype)
print(data1_x[32,2])


data2_x = torch.from_numpy(data) * 12.0
print(data2_x.shape)
print(data2_x.dtype)
print(data2_x[32,2])

criterion = torch.nn.MSELoss()
loss = criterion(data1_x, data2_x)
print(loss)
