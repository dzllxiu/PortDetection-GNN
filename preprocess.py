import numpy as np
from pathlib import Path
import json
from sklearn import preprocessing

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
            
if __name__ == '__main__':
    get_x_y()
            
            
            
        