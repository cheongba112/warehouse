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
