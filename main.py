import numpy as np
import argparse
import json
import torch.nn

def open_port_process(data):
	'''
	将开放端口数据转换为布尔型
	'''
	matrix_neighbor_y = []
	matrix_target_y = []
	for i in data['neighbor_y']: # 对每台主机，将65536个端口中开放的记为1，其余记为0
		port = np.zeros(65536)
		for j in i:
			if j != 0:
				port[int(j)] = 1
			else: break
		matrix_neighbor_y.append(port)

	for i in data['target_y']:
		port = np.zeros(65536)
		for j in i:
			if j != 0:
				port[int(j)] = 1
			else: break
		matrix_target_y.append(port)
	return matrix_neighbor_y, matrix_target_y


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--train_data', type=str, default='./data/graph_train.json')
	parser.add_argument('--test_data', type=str, default='./data/graph_test.json')

	parser.add_argument('--beta1', type=float, default=0.9)
	parser.add_argument('--beta2', type=float, default=0.999)
	parser.add_argument('--lambda1', type=float, default=7e-3)
	parser.add_argument('--lr', type=float, default=5e-3)
	parser.add_argument('--harved_epoch', type=int, default=5) 
	parser.add_argument('--early_stop_epoch', type=int, default=50)
	parser.add_argument('--saved_epoch', type=int, default=200)  
	args = parser.parse_args()

	Tensor = torch.FloatTensor

	model = GNN()
	model.apply(weights_init)
	lr = args.lr
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(args.beta1, args.beta2))
	torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

	with open(args.train_data, 'r') as f:
		train_data = json.load(f)
	with open(args.test_data, 'r') as f:
		test_data = json.load(f)

	port_matrix_neighbor, port_matrix_target = open_port_process(train_data) # 将开放端口数据转换为布尔矩阵

	# train
	for epoch in range(200):
		print('Epoch:', epoch)
		model.train()
		neighbor_x = Tensor(train_data['neighbor_x'])
		neighbor_y = Tensor(port_matrix_neighbor)
		target_x = Tensor(train_data['target_x'])
		target_y = Tensor(port_matrix_target)

		optimizer.zero_grad()
		output = model(neighbor_x, neighbor_y, target_x)