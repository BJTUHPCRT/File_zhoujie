from torch import nn

import torch
from sklearn import preprocessing

user_encoder = preprocessing.OneHotEncoder()
service_encoder = preprocessing.OneHotEncoder()



class User(nn.Module):
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
	def __init__(self, embedding_size, user_num=140, device="cpu"):
		super(User, self).__init__()
		self.device = device
		self.embedding_size = embedding_size
		self.user_num=user_num
		# one_hot编码过程
		self.user_matrix = self.load_user_msgs() # 140 * 236
		self.user_encode_elements = self.user_matrix.shape[1]
		# 自然语言处理上， embedding 的结果, id, city, AS => 3 * embedding_size
		# user * service => 6 * embedding_size
		# CNN => 卷积后的向量
		self.embedding_matrix = nn.Parameter(
			torch.randn((self.user_encode_elements, embedding_size)).to(device) # 236 * embedding_size
		)

	def load_user_msgs(self):
		user_target = []
		usermap = []
		# userlist 的embedding
		for line in open("./datas/userlist.txt", 'r'):
			arr = line.replace("\n", '').split("\t")
			id = int(arr[0])
			if id >= self.user_num:
				break
			# if arr[1] == 'None':
				# 需要占个位子，后面利用id找vecotr，本身并不使用
				# usermap.append(['0', "Worcester", '10326'])
			# else:
			tempt = [arr[0], arr[1], arr[4], arr[7].replace(" ", '')]
			usermap.append(tempt)
			user_target.append(tempt)
		user_encoder.fit(user_target)
		return torch.tensor(user_encoder.transform(usermap).toarray(), dtype=torch.float).to(self.device)

	def forward(self, data):
		# data : shape: batch_size
		# users = self.user_matrix[data]  # 用户的 batch_size * user_encode_num
		# if data == True:
		# 	return self.user_matrix.mm(self.embedding_matrix)
		users = self.user_matrix.index_select(0, data)  # 用户的 batch_size * user_encode_num
		return users.mm(self.embedding_matrix)
		# tempt = users > 0
		# batch_size = data.shape[0]
		# 我们需要做embedding结果得到的数据， batch_size * 3 * embedding_size
		# embedding_result_matrix = torch.zeros((batch_size, 3, self.embedding_size)).to(self.device)
		# for row_index in range(0, batch_size):
		# 	embedding_result_matrix[row_index] = self.embedding_matrix[tempt[row_index]]
		# 进行CNN 进行卷积
		# return embedding_result_matrix # batch_size * 3 * embedding_size





