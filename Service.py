
from torch import nn

import torch
import numpy as np
from torch.optim import Adam

from sklearn import preprocessing
# from datas.dataExplain import binary_search_user, binary_search_service

user_encoder = preprocessing.OneHotEncoder()
service_encoder = preprocessing.OneHotEncoder()


class Service(nn.Module):
	'''
		Conv2d 的输入维度:  batch_size * in_channels * weight * height
		Conv2d 的输出: batch_size * out_channels * （计算的weight * 计算的height)
		padding: （竖直方向的，水平方向的)
		kernel_size: (竖直方向的，水平方向的）
		model = nn.Conv2d(in_channels=1, out_channels=1, stride=1,kernel_size=(6,7),padding=(0, 3))
		input = torch.randn((1,1, 6, 20))
		res = model(input)
		print(res.shape)
	'''
	def __init__(self, embedding_size, device="cpu"):
		super(Service, self).__init__()
		self.device = device
		self.embedding_size = embedding_size
		# one_hot编码过程
		self.service_matrix = self.load_service_msgs()
		self.service_encode_elements = self.service_matrix.shape[1]
		# 自然语言处理上， embedding 的结果, id, city, AS => 3 * embedding_size
		# user * service => 6 * embedding_size
		# CNN => 卷积后的向量
		self.embedding_matrix = nn.Parameter(
			torch.randn((self.service_encode_elements, embedding_size)).to(device)
		)

	def load_service_msgs(self):
		# wslist的embedding
		service_target = []
		servicemap = []
		for line in open("./datas/wslist.txt", 'r'):
			arr = line.replace("\n", '').split("\t")
			# if arr[1] == 'None':
				# 需要占个位子，后面利用id找vecotr
				# servicemap.append(['0', 'Reno', '14627'])
			# else:
			tempt = [arr[0], arr[1], arr[4], arr[7].replace(" ", '')]
			service_target.append(tempt)
			servicemap.append(tempt)
		service_encoder.fit(service_target)
		return torch.tensor(service_encoder.transform(servicemap).toarray(), dtype=torch.float).to(self.device)

	def forward(self, data):
		# data : shape: batch_size
		# services = self.service_matrix[data]  # 用户的 batch_size * user_encode_num
		services = self.service_matrix.index_select(0, data)  # 用户的 batch_size * user_encode_num
		return services.mm(self.embedding_matrix)
		# tempt = services > 0
		# batch_size = data.shape[0]
		# # 我们需要做embedding结果得到的数据， batch_size * 3 * embedding_size
		# embedding_result_matrix = torch.zeros((batch_size, 3, self.embedding_size)).to(self.device)
		# for row_index in range(0, batch_size):
		# 	embedding_result_matrix[row_index] = self.embedding_matrix[tempt[row_index]]
		# # 进行CNN 进行卷积
		# return embedding_result_matrix  # batch_size * 3 * embedding_size
