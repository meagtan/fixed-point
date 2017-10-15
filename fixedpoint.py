import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FixedPoint:
	def __init__(self, f, Din, Dhid, Dout):
		self.f = f # f : R^Dout x R^Din -> R^Dout
		self.model = nn.Sequential(
			nn.Linear(Din, Dhid),
			nn.Tanh(),
			nn.Linear(Dhid, Dout)
		) # regression, no softmax layer
		self.l = nn.MSELoss()
	
	def loss(self, t, x):
		return self.l(x, f(x, t)) # f should apply to each row of x and t separately
	
	def forward(self, input):
		return self.model.forward(input)
	
	def train(self, t, eps):
		pass
