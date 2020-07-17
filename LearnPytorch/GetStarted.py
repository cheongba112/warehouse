# https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py

from __future__ import print_function
import torch
import numpy as np

# create tensor
# x = torch.empty(5, 3)                    # uninitialised
# x = torch.rand(5, 3)
# x = torch.zeros(5, 3, dtype=torch.long)  # set data type
x = torch.tensor([5.5, 3])

# create tensor based on an existing one
x = x.new_ones(5, 3, dtype=torch.double)    # new_* methods can set size
print(x)

x = torch.randn_like(x, dtype=torch.float)  # override dtype, but have same size
print(x)

# get size
print(x.size())  # torch.Size is in fact a tuple

# operations
# addition
y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)  # provide a tensor to store output
print(result)

y.add_(x)  # operation postfixed with '_' is in-place
print(y)

# numpy-like indexing
print(x[:, 1])

# reshape
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# get value from an one element tensor
x = torch.rand(1)
print(x)
print(x.item())

# numpy bridge
# convert a torch tensor to a numpy array
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

# b 'follows' a
a.add_(1)
print(a)
print(b)

# convert numpy array to torch tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# CUDA tensors
"""
# use 'torch.device()' objects to move tensors in and out of GPU
if torch.cuda.is_available():
	device = torch.device('cuda')
	y = torch.ones_like(x, device=device)
	x = x.to(device)
	z = x + y
	print(z)
	print(z.to('cpu', torch.double))
"""
