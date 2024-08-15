import numpy as np
from pathlib import Path
import json
from sklearn import preprocessing
import argparse
import pandas as pd

data_path = Path('./data/subnet')

def get_x_y():
	'''
	提取特征和开放端口信息, 特征包括asn, organization, ipv4地址的4个部分
	'''
	ports = []
	asn = []
	org = []
	ip_split1 = []
	ip_split2 = []
	ip_split3 = []
	ip_split4 = []
	for i in data_path.iterdir():
		f = open(i / 'infos.json', 'r')
		lines = f.readlines()
		for line in lines:
			temp = json.loads(line.strip())
			ports.append(temp['ports'])
			asn.append(temp['asn'])
			org.append(temp['organization'])
			ip_split1.append(temp['ip'].split('.')[0])
			ip_split2.append(temp['ip'].split('.')[1])
			ip_split3.append(temp['ip'].split('.')[2])
			ip_split4.append(temp['ip'].split('.')[3])
	
	asn = preprocessing.MinMaxScaler().fit_transform(np.array(asn).reshape(-1, 1)) # 一维数组转换为二维[num, 1]

	org = preprocessing.LabelEncoder().fit_transform(org)
	org = preprocessing.MinMaxScaler().fit_transform(np.array(org).reshape(-1, 1))

	ip_split1 = preprocessing.MinMaxScaler().fit_transform(np.array(ip_split1).reshape(-1, 1))
	ip_split2 = preprocessing.MinMaxScaler().fit_transform(np.array(ip_split2).reshape(-1, 1))
	ip_split3 = preprocessing.MinMaxScaler().fit_transform(np.array(ip_split3).reshape(-1, 1))
	ip_split4 = preprocessing.MinMaxScaler().fit_transform(np.array(ip_split4).reshape(-1, 1))

	x = np.concatenate((asn, org, ip_split1, ip_split2, ip_split3, ip_split4), axis=1)
	y = np.array(pd.DataFrame(ports))
	return x, y


def split_train_test(num, seed, train_test_ratio=0.8, neighbor_ratio=0.7):
	'''
	划分训练集和测试集
	'''
	np.random.seed(seed)
	index = list(range(num))
	np.random.shuffle(index) # 随机打乱下标
	neighbor_train = index[:int(num * train_test_ratio * neighbor_ratio)]
	target_train = index[int(num * train_test_ratio * neighbor_ratio):int(num * train_test_ratio)]
	neighbor_test = neighbor_train + target_train
	target_test = index[int(num * train_test_ratio):]
	return neighbor_train, target_train, neighbor_test, target_test

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_test_ratio', type=float, default=0.8)
    parser.add_argument('--neighbor_ratio', type=float, default=0.7)
    args = parser.parse_args()
    
    features, open_ports = get_x_y()
    
    neighbor_train, target_train, neighbor_test, target_test = split_train_test(features.shape[0], args.seed, args.train_test_ratio, args.neighbor_ratio)
            
            
            
        