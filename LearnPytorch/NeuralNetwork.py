import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):  # what is nn.Module?
	def __init__(self):
		super(Net, self).__init__()  # call __init__ in nn.Module?
		# 1 input image channel, 0 output channels, 3x3 square convolotion
		# what is channel?
		# kernel
		self.conv1 = nn.Conv2d(1, 6, 3)  # what ar the 3 parameters?
		self.conv2 = nn.Conv2d(6, 16, 3)
		# an affine operation: y = Wx + b
		self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		# max pooling over a (2, 2) window
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		# if the size is a square you can only specify single number
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features


net = Net()
print(net)


# have to define forward function, backword function is auto-defined using autograd

# learnable parameters, net.parameters()
params = list(net.parameters())
print(len(params))
print(params[0].size())


input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)


net.zero_grad()
out.backward(torch.randn(1, 10))


output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()        # Does the update
