import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
# from interaction_four_element import LSTM_Background
from interaction_four_element_origin import LSTM_Background
from torch.optim import Adam
from SCAGRU import SCAGRU
from utils import *
from config import *
from time import time
from TimeSeriesSimilarity import process

device = config["device"]
l2_lambda = config["l2_lambda"]
sigma = config["sigma"]
filled_matrix = config["fill_matrix_path"]
steps = config["steps"] # lstm时间步
batch_size = config["batch_size"]
epochs = config["epochs"]
embedding_middle_size = config["embedding_size"] # 我们用户服务的embedding长度
hidden_size = config["hidden_size"]  # lstm的隐藏向量长度
learning_rate = config["learning_rate"]
ratio = config["ratio"]

# invoke_matrix = torch.load("./targets/10percent/matrix_result_new_method.pt").to(device)
# user_similarity_matrix = torch.load("./targets/10percent/user_similarity.pt")
# service_similarity_matrix = torch.load("./targets/10percent/service_similarity.pt")
# user_similarity_matrix, service_similarity_matrix, invoke_matrix = process()


class PersonalizeLSTM(nn.Module):
	def __init__(self, embedding_middle_size, hidden_size, device, user_similarity_matrix, service_similarity_matrix,invoke_matrix, input_size = 1):
		super(PersonalizeLSTM, self).__init__()
		self.embedding_model = LSTM_Background(
			embedding_middle_size,
			hidden_size,
			user_similarity_matrix= user_similarity_matrix,
			service_similarity_matrix= service_similarity_matrix,
			device=device).to(device)
		self.invoke_matrix = invoke_matrix
		# self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size).to(device)
		self.gru = SCAGRU(indim = input_size, hidim = hidden_size, device=device).to(device)
		self.relu = nn.ReLU()
		self.linear_out = nn.Linear(hidden_size, input_size).to(device)
		self.device = device

	def forward(self, info_data):
		cur_batch_size = info_data.shape[0]
		# info:  batch_size * 3 (time, user_id, service_id)
		# response: batch_size
		# 构建时间序列
		input = torch.zeros(steps, cur_batch_size).to(device)
		step = 0  # 0 ,1 ,2, 3
		while step <= steps:
			tempt = self.invoke_matrix[info_data[:, 1], info_data[:, 2], (info_data[:, 0] - step) % 96]
			input[steps - 1 - step] = tempt
			step += 1
		input = input.unsqueeze(2)
		# data: seq_len * batch_size * input_size => 4 * 64 * 1
		# info_data: batch_size * 3 (time, user,service)
		# 手动的将 user * service 的交互结果带进来
		hidden_vecotr = self.embedding_model(info_data)
		# res = self.model(input, hidden_vecotr)
		res = self.gru(input, hidden_vecotr) # sca-gru
		res = self.relu(res)
		res = self.linear_out(res)
		res = torch.squeeze(res, 1).to(self.device)
		return res

def loss_fun(output, target):
	'''
		smooth l1 = {
			(f-y)^2,  |f-y| < sigma,
			2sigma|f - y| - sigma^2,  |f - y| > sigma
		}
	'''
	loss_type = config["loss_type"]  # 0: l1, 1: l2, 2: l1 + l2, 3: smooth l1
	if (loss_type == 0):
		return mae_loss(output, target)
	elif (loss_type == 1):
		return rmse_loss(output, target)
	elif (loss_type == 3):
		return huber_loss(output, target)

def load_data():
	# 读取数据
	train_info_data, train_rt_data, test_info_data, test_rt_data, train_matrix = load_data_split("./datas/rtdata.csv", ratio / 100)
	train_num = train_info_data.shape[0]
	train_dataset = TensorDataset(train_info_data, train_rt_data)
	train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
	test_num = test_info_data.shape[0]
	test_dataset = TensorDataset(test_info_data, test_rt_data)
	test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size * 30)
	return train_dataloader, train_num, test_dataloader, test_num, train_matrix

def test (model, test_dataloader, test_num, output_file):
	test_rmse_loss = 0
	test_mae_loss = 0
	start = time()
	with torch.no_grad():
		for info, response in test_dataloader:
			info = info.to(device)
			response = response.reshape(-1).to(device)
			res = model(info)
			test_rmse_loss += rmse_loss(res, response).item()
			test_mae_loss += mae_loss(res, response).item()
		res_str = ">>>>>>>> test result: cost %d, test rmse loss is %.5f, mae loss is %.5f \n" % \
		          (time() - start, (test_rmse_loss / test_num) ** 0.5, (test_mae_loss / test_num))
		output_file.write(res_str)
		output_file.flush()

def train():
	# 输出文件
	outputfilename = config["outputfilename"]
	result = open(outputfilename, 'a')
	result.write("\n")
	result.flush()
	train_dataloader, train_num, test_dataloader, test_num, train_matrix = load_data()
	user_similarity_matrix, service_similarity_matrix = process(train_matrix)
	# model
	model = PersonalizeLSTM(
		embedding_middle_size, hidden_size, device,
		user_similarity_matrix=user_similarity_matrix.to(device),
		service_similarity_matrix= service_similarity_matrix.to(device),
		invoke_matrix=train_matrix.to(device)
	).to(device)
	adam = Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
	for epoch in range(epochs):
		start = time()
		print("epoch", epoch, "start")
		train_rmse_loss = 0
		train_mae_loss = 0
		# 训练
		for info, response in train_dataloader:
			info = info.to(device)
			response = response.reshape(-1).to(device)
			res = model(info)
			adam.zero_grad()
			loss = loss_fun(res, response)
			train_rmse_loss += rmse_loss(res, response)
			train_mae_loss += mae_loss(res, response)
			loss.backward()
			adam.step()
		cost = time() - start
		epoch_status = "epoch %d train cost time %.2f, rmse loss is %.5f, mae loss is %.5f \n"%\
		             (epoch, cost, torch.sqrt(train_rmse_loss / train_num).item(), (train_mae_loss / train_num).item())
		result.write(epoch_status)
		result.flush()
		if epoch != 0 and epoch % 3 == 0:
			test(model, test_dataloader, test_num, result)
	result.close()

if __name__ == "__main__":
	train()









