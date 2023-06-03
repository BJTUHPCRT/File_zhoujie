
# 没有数据 1
# 没有数据 不计入， 算平均

# 64 * 0.05 = 3个
# 32 * 0.2 = 6.4个
# 均一化 => 0代表这个时间网络好， => 1 代表这个时间网络一般

import torch
from torch import nn
import heapq
from time import time
from datas.dataExplain import *
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
from config import *

user_num = config['user_num']
service_num = config['service_num']
time_slots = config['time_slots']
ratio = config['ratio']

# 记录范围值的
invoke_records = torch.zeros(user_num, service_num, time_slots, dtype=torch.float)
invoke_range = torch.zeros(user_num, service_num, 2, dtype=torch.float)
invoke_count = torch.zeros(user_num, service_num, dtype=torch.int)
# 记录一下可能唯一调用时间片
invoke_only_once = torch.ones(user_num, service_num, dtype=torch.int)
# 记录一下用户的调用记录范围
invoke_user_range = torch.zeros(user_num, 2)
# 用户的时间特性tensor
user_time_status = torch.zeros(user_num, time_slots, dtype=torch.float)
service_time_status = torch.zeros(service_num, time_slots, dtype=torch.float)
# 用户间的相似度
user_similarity_matrix = torch.zeros(user_num, user_num, dtype=torch.float)
service_similarity_matrix = torch.zeros(service_num, service_num, dtype=torch.float)


def minMaxScaler(data):
	size = data.shape[0]
	min = torch.min(data)
	max = torch.max(data)
	if min == max:
		return torch.ones(size)
	return (data - min) / (max - min) + 0.0001

def data_handle(invoke_real_records):
	global invoke_range
	global invoke_count
	global invoke_only_once
	global invoke_user_range
	for user_id in range(0, user_num):
		# if binary_search_user(user_id):
		# 	continue
		for service_id in range(0, service_num):
			# if binary_search_service(service_id):
			# 	continue
			invokes = invoke_real_records[user_id][service_id]
			indexs = (invokes > 0).nonzero().reshape(-1)  # time_slots
			count = indexs.shape[0]
			if count == 0:
				continue
			invoke_count[user_id][service_id] = count
			invoke_positives = invokes[indexs]
			max = invoke_positives.max() # 最大值
			min = invoke_positives.min() # 最小值
			invoke_range[user_id][service_id][0] = min
			invoke_range[user_id][service_id][1] = max
			invoke_only_once[user_id][service_id] = indexs[count - 1]
			# 计算用户的调用记录范围
			cur_user_min = invoke_user_range[user_id][0].item()
			cur_user_max = invoke_user_range[user_id][1].item()
			invoke_user_range[user_id][0] = min if cur_user_min > min or cur_user_min==0 else cur_user_min
			invoke_user_range[user_id][1] = max if cur_user_max < max else cur_user_max
			range_scope = minMaxScaler(invoke_positives)
			if count == 1:
				invoke_records[user_id][service_id][indexs[0]] = 0.5
			else:
				invoke_records[user_id][service_id][indexs] = range_scope


def calc_user_time_status():
	for user_id in range(0, user_num):
		# if binary_search_user(user_id):
		# 	continue
		for time_id in range(0, time_slots):
			record = invoke_records[user_id,:,time_id]
			record = record[record > 0]
			user_time_status[user_id][time_id] = 0 if record.shape[0] == 0 else record.mean().item()

def calc_service_time_status():
	for service_id in range(0, service_num):
		# if binary_search_service(service_id):
		# 	continue
		for time_id in range(0, time_slots):
			record = invoke_records[:,service_id,time_id]
			record = record[record > 0]
			service_time_status[service_id][time_id] = 0 if record.shape[0] == 0 else record.mean().item()

# 计算两个用户之间的相似度
# 所有对应时间片的平均值
def calc_user_similarity(user_i, user_j):
	return 1 / (1 + torch.pow(user_time_status[user_i] - user_time_status[user_j], 2).sum())
	# return torch.cosine_similarity(user_time_status[user_i], user_time_status[user_j], dim=0)

def calc_all_users_similarity():
	for cur in range(0, user_num):
		# if (binary_search_user(cur)):
		# 	continue
		for next in range(cur + 1, user_num):
			# if (binary_search_user(next)):
			# 	continue
			sim = calc_user_similarity(cur, next)
			user_similarity_matrix[cur][next] = sim
			user_similarity_matrix[next][cur] = sim
	# torch.save(user_similarity_matrix, "./targets/%s/user_similarity.pt"%ratio_str)

def calc_service_similarity(service_i, service_j):
	return 1 / (1 + torch.pow(service_time_status[service_i] - service_time_status[service_j], 2).sum())
	# return torch.cosine_similarity(service_time_status[service_i], service_time_status[service_j], dim=0)

def calc_all_services_similarity():
	for cur in range(0, service_num):
		# if (binary_search_service(cur)):
		# 	continue
		for next in range(cur + 1, service_num):
			# if (binary_search_service(next)):
			# 	continue
			sim = calc_service_similarity(cur, next)
			service_similarity_matrix[cur][next] = sim
			service_similarity_matrix[next][cur] = sim
	# torch.save(service_similarity_matrix, "./targets/%s/service_similarity.pt"%ratio_str)

# 计算相似用户
def cal_similar_user_map():
	similar_users_map = {}
	for user_id in range(0, user_num):
		# if binary_search_user(user_id):
		# 	continue
		tempt = [] # 得到相似向量
		for i in range(0, user_num):
			# if binary_search_user(i):
			# 	continue
			# 找到用户间相似度大于 0 的记录到相似用户中
			similar = user_similarity_matrix[user_id][i]
			if similar > 0 and i!=user_id:
				tempt.append(i)
			else:
				pass
		similar_users_map[user_id] = tempt
	return similar_users_map

# 填充矩阵
# 如果没有相似用户怎么办，在当前时间下的所有均值，当前用户的均值
def fill_matrix(invoke_real_records):
	similar_user_map = cal_similar_user_map() # 所有的相似用户id
	# 140 * 1000 * 96 时间复杂度不高啊
	for user_id in range(0, user_num):
		# if binary_search_user(user_id):
		# 	continue
		users = similar_user_map[user_id]
		users = torch.tensor(users) # 100
		if len(users) == 0:
			continue
		if user_id % 30 == 0:
			print(user_id, "user_id enter")
		tempt = invoke_records[users, :, :] # 100 * 1000 * 96
		similars = user_similarity_matrix[user_id][users].unsqueeze(0) # [1 * 100]
		total_similar = similars.sum().item()
		# 一、res != 0
		# 	1. 记录数量 > 2, 有范围，可以直接计算
		# 	2. 记录数量 == 1, 根据用户有数据的,计算倍数
		# 	3. 记录数量 == 0，没有记录，就用当前用户的 范围，计算
		# 	invoke_records[user_id][service_id][time_slot] = res
		# 二、res == 0
		for service_id in range(0, service_num):
			# if binary_search_service(service_id):
			# 	continue
			invoke_num = invoke_count[user_id][service_id] # 调用数量
			user_time_matrix = tempt[:, service_id, :] # 100 * 96
			target = similars.mm(user_time_matrix).squeeze(0) # 1 * 96 => 96
			target = target / total_similar
			time_ids = target == 0 # 找到相似用户没有调用过的时间片, 使用用户的平均
			target[time_ids] = user_time_status[user_id][time_ids]
			times_need_fill = invoke_real_records[user_id][service_id] == 0 # 需要填充数据的时间片们
			if invoke_num >= 2:
				target = (invoke_range[user_id][service_id][1] - invoke_range[user_id][service_id][0]) * target + invoke_range[user_id][service_id][0]
			elif invoke_num == 1:
				time = invoke_only_once[user_id][service_id].item()
				status = 0.5 if user_time_status[user_id][time] == 0 else user_time_status[user_id][time]
				target = (user_time_status[user_id] / status) * invoke_real_records[user_id][service_id][time] # tensor 96
			else:
				target = (invoke_user_range[user_id][1] - invoke_user_range[user_id][0]) * target + invoke_user_range[user_id][0]
			invoke_real_records[user_id][service_id][times_need_fill] = target[times_need_fill] # 只填充没有的数据

def process(train_matrix):
	# 加载处理数据
	data_handle(train_matrix)
	# 计算相似度
	calc_user_time_status()
	calc_all_users_similarity()

	calc_service_time_status()
	calc_all_services_similarity()

	calc_all_services_similarity()

	fill_matrix(train_matrix)

	return user_similarity_matrix, service_similarity_matrix

