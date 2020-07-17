import torch
import torch.nn as nn
import torch.nn.Functional as F


# way 1 to use GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = x.to(device)
y = y.to(device)


# way 2 to use GPU
model = model.cuda()
x = x.cuda()
y = y.cuda()


# data parallelism
'''
Data Parallelism is when we split the mini-batch of samples into multiple
smaller mini-batches and run the computation for each of the smaller
mini-batches in parallel.
'''
