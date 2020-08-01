import torch
import torch.nn as nn

class net(nn.Module):
	def __init__(self):
			super(net, self).__init__()
			self.seq = nn.Sequential(
				nn.Linear(8, 6),
				nn.ReLU(),
				nn.Linear(6, 4),
				nn.Sigmoid(),
				nn.Linear(4, 1)
			)

	def forward(self, x):
		return self.seq(x)


if __name__ == '__main__':
	n = net()
	
	optimizer = torch.optim.SGD(n.parameters(), lr=0.02)
	loss_func = nn.MSELoss()

	x = torch.Tensor([
		[2.,4.,2.,3.,5.,7.,7.,5.],
		[5.,2.,1.,1.,2.,4.,6.,8.],
		[9.,7.,4.,2.,1.,4.,6.,8.],
		[6.,4.,2.,1.,3.,5.,7.,8.],
		[8.,6.,4.,1.,2.,3.,5.,7.],
		[9.,8.,5.,3.,3.,2.,3.,5.],
		[1.,4.,3.,3.,5.,8.,9.,9.],
		[0.,9.,7.,4.,7.,8.,5.,4.],
		[6.,4.,2.,2.,2.,4.,6.,7.],
		[6.,1.,3.,5.,7.,9.,8.,4.],
		])
	y = torch.Tensor([  # y = sum(x) % 4
		[3.],
		[1.],
		[1.],
		[0.],
		[0.],
		[2.],
		[2.],
		[0.],
		[1.],
		[3.],
		])

	print(x)
	print(y)

	for epoch in range(1000):
		pred = n(x)
		loss = loss_func(pred, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# if not epoch % 100:
		# 	print(pred, loss)

	for p in pred:
		print(round(p.tolist()[0]))
