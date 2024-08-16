import numpy as np
import argparse
import json

def open_port_process(data):
	'''
	将开放端口数据转换为布尔型
	'''
	matrix = []
	for i in data['neighbor_y']: # 对每台主机，将65536个端口中开放的记为1，其余记为0
		port = np.zeros(65536)
		for j in i:
			if j != 0:
				port[int(j)] = 1
			else: break
		matrix.append(port)
	return matrix


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--train_data', type=str, default='./data/graph_train.json')
	parser.add_argument('--test_data', type=str, default='./data/graph_test.json')
	args = parser.parse_args()

	with open(args.train_data, 'r') as f:
		train_data = json.load(f)
	with open(args.test_data, 'r') as f:
		test_data = json.load(f)

	port_matrix = open_port_process(train_data) # 将开放端口数据转换为布尔矩阵
	