import torch
from torch import nn
from torch.nn import Module
from User import User
from Service import Service
from config import *
from time import time

user_num = config["user_num"]
service_num = config["service_num"]
time_slots = config["time_slots"]
similar_num = config["similar_num"]



class Interaction(Module):
	'''
	输入是batch_size * k * embedding_size, 我们需要做的就是拆分成[ batch_index, 0] [batch_index, 1] 这类的样子
	'''
	def __init__(self, input_size, device='cpu'):
		super(Interaction, self).__init__()
		self.device=device
		self.input_size = input_size
		self.interaction = nn.Linear(input_size, input_size).to(device)
		self.forget_gate = nn.Linear(input_size, input_size).to(device)

	def forward(self, datas):
		'''
		这个输入向量，维度应该是  batch_size * 4 * hidden_size
		C_3^2 => 3 个 二阶交互 + 一阶信息
		4 ^ 2 => 6
		'''
		# a = datas[:, 0, :]
		# b = datas[:, 1, :]
		a = datas[0]
		b = datas[1]
		forget_rate = torch.sigmoid(self.forget_gate(a + b)).to(self.device) # batch_size * input_size
		tempt = self.interaction(a + b) # batch_size * input_size
		return forget_rate * tempt + (1 - forget_rate) * (a + b)


class Pair_Interaction_Modify(Module):
	'''
	可能要有所改动，pmln的代码是一个model集成了所有的交互动作，这样确实也能减少模型的训练量
	不然同时留个模型确实太大了，可以进行改进的
	'''
	def __init__(self, input_size, output_size, interaction_times = 1, device='cpu'):
		super(Pair_Interaction_Modify, self).__init__()
		self.input_size =  input_size
		self.fc = nn.Linear(input_size * 4, output_size).to(device)
		self.interaction_times = interaction_times
		self.device = device
		self.model_1 = nn.Sequential(
			Interaction(input_size, device),
			nn.ReLU()
		).to(device)
		self.model_2 = nn.Sequential(
			Interaction(input_size, device),
			nn.ReLU()
		).to(device)
		# self.tempt1 = torch.zeros(256, 6, self.input_size).to(device)
		# self.tempt2 = torch.zeros(256, 6, self.input_size).to(device)
		# self.result1 = torch.zeros(256, 4, self.input_size).to(device)
		# self.result2 = torch.zeros(256, 4, self.input_size).to(device)

	def forward(self, datas):
		'''
		这个输入向量，维度应该是 (a,b,c,d)  batch_size * hidden_size
		C_3^2 => 3 个 二阶交互 + 一阶信息
		4 ^ 2 => 6
		'''
		# datas 的shape: batch_size * k * input_size
		batch_size = datas[0].shape[0]
		tempt1 = torch.zeros(batch_size, 6, self.input_size).to(self.device)
		result1 = torch.zeros(batch_size, 4, self.input_size).to(self.device)
		# tempt2 = torch.zeros(batch_size, 6, self.input_size).to(self.device)
		# result2 = torch.zeros(batch_size, 4, self.input_size).to(self.device)
		count = 0
		for i in range(0, 4):
			for j in range(i + 1, 4):
				tempt1[:, count, :] = self.model_1((datas[i], datas[j]))  # batch_size * input_size
				count += 1
		norms1 = torch.norm(tempt1, dim=2)  # 选择最大的 batch_size * geinerate_count
		_, indexs1 = norms1.topk(4, dim=1)  # batch_size * 4
		for i in range(0, batch_size):
			try:
				result1[i] = tempt1[i].index_select(0, indexs1[i])
			except Exception as e:
				print(tempt1, indexs1, e)

		#第二轮的交互，可以考虑干掉
		# count = 0
		# for i in range(0, 4):
		# 	for j in range(i + 1, 4):
		# 		tempt2[:, count, :] = self.model_2(
		# 			(result1[:, i, :], result1[:, j, :]))  # batch_size * input_size
		# 		count += 1
		# norms2 = torch.norm(tempt2, dim=2)  # 选择最的 batch_size * geinerate_count
		# _, indexs2 = norms2.topk(4, dim=1)  # batch_size * 4
		# for i in range(0, batch_size):
		# 	result2[i] = tempt2[i].index_select(0, indexs2[i])
		input = torch.cat((result1[:, 0, :],
		                   result1[:, 1, :],
		                   result1[:, 2, :],
		                   result1[:, 3, :]), dim=1)
		return self.fc(input)



class LSTM_Background(Module):
	def __init__(self, hidden_size, output_size, user_similarity_matrix, service_similarity_matrix, device='cpu'):
		super(LSTM_Background, self).__init__()
		self.user_similarity_matrix = user_similarity_matrix.to(device)
		self.service_similarity_matrix = service_similarity_matrix.to(device)
		self.device = device
		self.similar_user_num = similar_num
		self.similar_service_num = similar_num
		self.hidden_size = hidden_size
		self.user_model = User(hidden_size, user_num,device)
		self.service_model = Service(hidden_size, device)
		self.similar_user_map = self.cal_similar_user_map()
		self.similar_service_map = self.cal_similar_service_map()
		self.all_users = torch.LongTensor([k for k in range(user_num)]).to(self.device)
		self.all_service = torch.LongTensor([k for k in range(service_num)]).to(self.device)
		self.cnn = nn.Conv2d(
			in_channels=1,
			out_channels=1,
			stride=1,
			kernel_size=(similar_num, 7),
			padding=(0, 3)
		).to(device)
		self.service_cnn = nn.Conv2d(
			in_channels=1,
			out_channels=1,
			stride=1,
			kernel_size=(similar_num, 7),
			padding=(0, 3)
		).to(device)
		self.pair_interaction = Pair_Interaction_Modify(hidden_size, output_size, device=device).to(device)

	# 计算相似用户
	def cal_similar_user_map(self):
		similar_users_map = {}
		for user_id in range(0, user_num):
			# 计算某个用户的相似用户 TopK个
			_, indexs = self.user_similarity_matrix[user_id].topk(self.similar_user_num)
			tempt, _ = indexs.sort()
			similar_users_map[user_id] = tempt.tolist()
		return similar_users_map

		# 计算相似用户
	def cal_similar_service_map(self):
		similar_service_map = {}
		for service_id in range(0, service_num):
			# 计算某个用户的相似用户 TopK个
			_, indexs = self.service_similarity_matrix[service_id].topk(self.similar_service_num)
			tempt, _ = indexs.sort()
			similar_service_map[service_id] = tempt.tolist()
		return similar_service_map

	def forward(self, data):
		# data shape: batch_size * 4 (time_id, user_id, service_id, rt)
		# 1. 求相似用户
		# 2. 求用户的向量
		# 3. 求服务的向量
		# 4. 相似用户的卷积
		# 5. 三个向量的合并
		batch_size = data.shape[0]
		# user_vectors = self.user_model(data[:, 1]) #batch_size * hidden_size
		user_ids = data[:, 1].tolist() #[id1,id2,....,idn]
		service_ids = data[:, 2].tolist()
		a = self.user_model(data[:, 1]) #batch_size * hidden_size
		b = self.service_model(data[:, 2]) # batch_size * hidden_size
		# 下面是相似用户
		user_neighbors = torch.LongTensor([self.similar_user_map[user_id] for user_id in user_ids]).to(self.device) # batch_size * k
		user_neighbors = user_neighbors.reshape(-1) # batch_size ✖️ times = total
		# total * hidden_size => batch_size * 1 * k * hidden_size => batch_size * hidden_size
		c = self.cnn(self.user_model(user_neighbors).reshape(-1, 1, self.similar_user_num, self.hidden_size)).reshape(batch_size, -1) # bach_size * k

		service_neighbors = torch.LongTensor([self.similar_service_map[service_id] for service_id in service_ids]).to(self.device) # batch_size * k
		service_neighbors = service_neighbors.reshape(-1)
		d = self.service_cnn(self.service_model(service_neighbors).reshape(-1, 1, self.similar_service_num, self.hidden_size)).reshape(batch_size, -1)

		res = self.pair_interaction((a,b,c,d)) # batch_size * output_size
		return res

# device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# model = LSTM_Background(64, 64,k=4, similar_service_num=4, device=device)
# input = torch.LongTensor([
# 	[4, 0, 0],
# 	[33, 0, 0],
# 	[37, 0, 0],
# 	[75, 0, 0],
# 	[80, 0, 0],
# 	[64, 0, 1],
# 	[89, 0, 1],
# 	[90, 0, 1],
# 	[46, 0, 2],
# 	[74, 0, 2],
# 	[84, 0, 2],
# 	[3, 0, 3],
# 	[41, 0, 3],
# 	[68, 0, 3],
# 	[4, 0, 4],
# 	[17, 0, 4],
# 	[23, 0, 4],
# 	[70, 0, 4],
# 	[93, 0, 4],
# 	[27, 0, 5],
# 	[35, 0, 5],
# 	[44, 0, 5],
# 	[58, 0, 5],
# 	[13, 0, 6],
# 	[36, 0, 6],
# 	[39, 0, 6],
# 	[69, 0, 6],
# 	[80, 0, 6],
# 	[91, 0, 6],
# 	[5, 0, 7],
# 	[6, 0, 7],
# 	[30, 0, 7],
# 	[52, 0, 7],
# 	[67, 0, 7],
# 	[85, 0, 7],
# 	[16, 0, 8],
# 	[81, 0, 8 ],
# 	[5, 0, 9],
# 	[14, 0, 9],
# 	[30, 0, 9],
# 	[37, 0, 9],
# 	[78, 0, 9],
# 	[23, 0, 11],
# 	[34, 0, 11],
# 	[36, 0, 11],
# 	[62, 0, 11],
# 	[37, 0, 12],
# 	[78, 0, 12],
# 	[79, 0, 12],
# 	[93, 0, 12],
# 	[34, 0, 13],
# 	[43, 0, 13],
# 	[61, 0, 13],
# 	[78, 0, 13],
# 	[89, 0, 13],
# 	[23, 0, 14],
# 	[31, 0, 14],
# 	[36, 0, 14],
# 	[67, 0, 14],
# 	[71, 0, 14],
# 	[74, 0, 14],
# 	[13, 0, 15],
# 	[27, 0, 15],
# 	[37, 0, 15],
# 	[55, 0, 15],
# 	[77, 0, 15],
# 	[90, 0, 15],
# 	[32, 0, 21],
# 	[41, 0, 21],
# 	[48, 0, 21],
# 	[50, 0, 21],
# 	[64, 0, 21],
# 	[9, 0, 22],
# 	[70, 0, 22],
# 	[88, 0, 22],
# 	[25, 0, 23],
# 	[50, 0, 23],
# 	[53, 0, 23],
# 	[61, 0, 23],
# 	[85, 0, 23],
# 	[24, 0, 24],
# 	[36, 0, 24],
# 	[42, 0, 24],
# 	[35, 0, 27],
# 	[74, 0, 27],
# 	[5, 0, 28],
# 	[48, 0, 28],
# 	[51, 0, 28],
# 	[91, 0, 28],
# 	[0, 0, 29],
# 	[3, 0, 29],
# 	[18, 0, 29],
# 	[48, 0, 29],
# 	[53, 0, 29],
# 	[77, 0, 29],
# 	[26, 0, 30],
# 	[52, 0, 30],
# 	[64, 0, 30],
# 	[65, 0, 30],
# 	[4, 0, 31],
# 	[23, 0, 31],
# 	[29, 0, 31],
# 	[38, 0, 31],
# 	[44, 0, 31],
# 	[46, 0, 31],
# 	[67, 0, 31],
# 	[74, 0, 31],
# 	[17, 0, 32],
# 	[19, 0, 32],
# 	[20, 0, 32],
# 	[58, 0, 32],
# 	[76, 0, 32],
# 	[85, 0, 32],
# 	[34, 0, 33],
# 	[62, 0, 33],
# 	[65, 0, 33],
# 	[14, 0, 34],
# 	[55, 0, 34],
# 	[41, 0, 35],
# 	[5, 0, 36],
# 	[11, 0, 36],
# 	[36, 0, 36],
# 	[73, 0, 36],
# 	[3, 0, 37],
# 	[34, 0, 37],
# 	[83, 0, 37],
# 	[5, 0, 38],
# 	[0, 0, 40],
# 	[1, 0, 40],
# 	[60, 0, 40],
# 	[64, 0, 40],
# 	[12, 0, 41],
# 	[18, 0, 41],
# 	[59, 0, 41],
# 	[67, 0, 41],
# 	[76, 0, 41],
# 	[10, 0, 42],
# 	[32, 0, 42],
# 	[42, 0, 42],
# 	[49, 0, 42],
# 	[54, 0, 42],
# 	[57, 0, 42],
# 	[56, 0, 43],
# 	[79, 0, 43],
# 	[86, 0, 43],
# 	[47, 0, 44],
# 	[62, 0, 44],
# 	[63, 0, 44],
# 	[82, 0, 44],
# 	[16, 0, 45],
# 	[82, 0, 45],
# 	[4, 0, 46],
# 	[17, 0, 46],
# 	[89, 0, 46],
# 	[93, 0, 46],
# 	[66, 0, 47],
# 	[69, 0, 47],
# 	[9, 0, 48],
# 	[12, 0, 48],
# 	[15, 0, 48],
# 	[46, 0, 48],
# 	[66, 0, 48],
# 	[79, 0, 48],
# 	[2, 0, 49],
# 	[9, 0, 49],
# 	[43, 0, 49],
# 	[2, 0, 50],
# 	[49, 0, 50],
# 	[50, 0, 50],
# 	[64, 0, 50],
# 	[18, 0, 51],
# 	[37, 0, 51],
# 	[64, 0, 51],
# 	[71, 0, 51],
# 	[75, 0, 51],
# 	[79, 0, 51],
# 	[7, 0, 52],
# 	[39, 0, 52],
# 	[54, 0, 52],
# 	[61, 0, 52],
# 	[26, 0, 56],
# 	[35, 0, 56],
# 	[41, 0, 56],
# 	[42, 0, 56],
# 	[61, 0, 56],
# 	[29, 0, 57],
# 	[30, 0, 57],
# 	[40, 0, 57],
# 	[50, 0, 57],
# 	[55, 0, 57],
# 	[67, 0, 57],
# 	[70, 0, 57],
# 	[93, 0, 57],
# 	[9, 0, 58],
# 	[20, 0, 58],
# 	[30, 0, 58],
# 	[53, 0, 58],
# 	[10, 0, 59],
# 	[68, 0, 59],
# 	[70, 0, 59],
# 	[79, 0, 59],
# 	[18, 0, 61],
# 	[36, 0, 61],
# 	[41, 0, 61],
# 	[3, 0, 62],
# 	[8, 0, 62],
# 	[13, 0, 62],
# 	[24, 0, 62],
# 	[34, 0, 62],
# 	[45, 0, 62],
# 	[59, 0, 62],
# 	[64, 0, 62],
# 	[92, 0, 62],
# 	[35, 0, 63],
# 	[89, 0, 63],
# 	[7, 0, 64],
# 	[20, 0, 64],
# 	[38, 0, 64],
# 	[81, 0, 64],
# 	[90, 0, 64],
# 	[20, 0, 65],
# 	[32, 0, 65],
# 	[43, 0, 74],
# 	[50, 0, 74],
# 	[7, 0, 75],
# 	[11, 0, 75],
# 	[22, 0, 75],
# 	[35, 0, 75],
# 	[36, 0, 75],
# 	[43, 0, 75],
# 	[45, 0, 75],
# 	[71, 0, 75],
# 	[5, 0, 77],
# 	[26, 0, 77],
# 	[28, 0, 77],
# 	[7, 0, 78],
# 	[25, 0, 78],
# 	[34, 0, 78],
# 	[48, 0, 78],
# 	[63, 0, 78],
# 	[68, 0, 78],
# 	[2, 0, 79],
# 	[6, 0, 79],
# ]).to(device)
# start = time()
# output = model(input)
# print(time() - start, "costed")
#
# start = time()
# output = model(input)
#
# print(time() - start, "costed")
#
# start = time()
# output = model(input)
#
# print(time() - start, "costed")