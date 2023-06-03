import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SCAGRU(nn.Module):
	def __init__(self, indim, hidim, device):
		super(SCAGRU, self).__init__()
		self.indim = indim
		self.hidim = hidim
		self.device = device
		self.W_zh, self.W_zx, self.b_z = self.get_three_parameters()
		self.W_rh, self.W_rx, self.b_r = self.get_three_parameters()
		self.W_hh, self.W_hx, self.b_h = self.get_three_parameters()
		self.W_ch, _, self.b_c = self.get_three_parameters()
		self.reset()

	def forward(self, input, state):
		input = input.type(torch.float32)
		# Y = []
		h = state
		c = F.relu(state @ self.W_ch + self.b_c)
		for x in input:
			z = torch.sigmoid(h @ self.W_zh + x @ self.W_zx + self.b_z + c)
			r = torch.sigmoid(h @ self.W_rh + x @ self.W_rx + self.b_r + c)
			ht = torch.tanh((h * r) @ self.W_hh + x @ self.W_hx + self.b_h + c)
			h = (1 - z) * h + z * ht
			# y = self.Linear(h)
			# Y.append(y)
		return h

	def get_three_parameters(self):
		indim, hidim = self.indim, self.hidim
		return nn.Parameter(torch.FloatTensor(hidim, hidim)).to(device=self.device), \
		       nn.Parameter(torch.FloatTensor(indim, hidim)).to(device=self.device), \
		       nn.Parameter(torch.FloatTensor(hidim)).to(device=self.device)

	def reset(self):
		stdv = 1.0 / math.sqrt(self.hidim)
		for param in self.parameters():
			nn.init.uniform_(param, -stdv, stdv)


