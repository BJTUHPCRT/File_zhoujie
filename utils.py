from random import random
import torch
from config import *

device = config["device"]
loss_type = config["loss_type"]
huber_sigma = config["sigma"]
user_num = config["user_num"]
service_num = config["service_num"]
time_slots = config["time_slots"]

def load_data_split(data_path, ratio):
	train_info = []
	train_rt = []
	test_info = []
	test_rt = []
	train_matrix = torch.zeros(user_num, service_num, time_slots, dtype=torch.float)
	for line in open(data_path, encoding="utf-8"):
		arr = line.replace("\t", '').split(" ")
		info = [int(arr[0]), int(arr[1]), int(arr[2])] # time, user, service
		rt = [float(arr[3])]
		if random() < ratio:
			train_info.append(info)
			train_rt.append(rt)
			train_matrix[info[1]][info[2]][info[0]] = rt[0]
		else:
			test_info.append(info)
			test_rt.append(rt)
	return torch.tensor(train_info).to(device),\
	       torch.tensor(train_rt).to(device),\
	       torch.tensor(test_info).to(device),\
	       torch.tensor(test_rt).to(device),\
		   train_matrix

def rmse_loss(output, target):
	return torch.pow(output - target, 2).sum()

def mae_loss(output, target):
	return torch.abs(output - target).sum()

def huber_loss(output, target):
	'''
		smooth l1 = {
			(f-y)^2,  |f-y| < sigma,
			2sigma|f - y| - sigma^2,  |f - y| > sigma
		}
	'''
	tempt = output - target
	t1 = tempt * tempt
	t2 = torch.abs(tempt) * 2 * huber_sigma - huber_sigma * huber_sigma
	res = torch.where(tempt.abs() <= huber_sigma, t1, t2)  # <1 => l2, >1 => l1
	return res.sum()

