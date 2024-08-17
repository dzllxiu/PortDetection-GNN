import torch
import torch.nn as nn

class GNN(nn.Module):
	def __init__(self, dim_in):
		super(GNN, self).__init__()
		self.dim_in = dim_in
		self.dim_z = dim_in + 65536
        
		self.att_similarity = SimpleAttentionLayer(temperature = self.dim_z ** 0.5,
					     d_q_in = self.dim_in,
						 d_k_in = self.dim_in,
						 d_v_in = self.dim_z,
						 d_q_out = self.dim_z,
						 d_k_out = self.dim_z,
						 d_v_out = self.dim_z)
		
		self.w = nn.Linear(self.dim_z, self.dim_z)
		self.output_layer = nn.Linear(self.dim_z, 65536) # 输出层，将dim_z维的向量映射到65536维的向量

	
	def forward(self, neighbor_x, neighbor_y, target_x):
		# neighbor_x: (batch_size, dim_in)
		# neighbor_y: (batch_size, 65536)
		# target_x: (batch_size, dim_in)
		# target_y: (batch_size, 65536)
		# output: (batch_size, 65536)
		n_neighbor = neighbor_x.size(0)
		n_target = target_x.size(0)
		ones = torch.ones(n_neighbor + n_target)
		neighbor_feature = torch.cat([neighbor_x, neighbor_y], dim=1)
		target_feature = torch.cat([target_x, torch.zeros(n_target, 65536)], dim=1) # target_y is not known
		all_feature = torch.cat([neighbor_feature, target_feature], dim=0)

		adj_matrix = torch.diag(ones) # (n_neighbor + n_target, n_neighbor + n_target)

		# compute attention
		_, similarity = self.att_similarity(neighbor_x, target_x, neighbor_feature)
		similarity = torch.exp(similarity)

		# update adj_matrix
		adj_matrix[n_neighbor:, :n_neighbor] = similarity

		degree = torch.sum(adj_matrix, dim=1)
		degree_reverse = 1.0 / (degree + 1e-12)
		degree_reverse = torch.diag(degree_reverse)

		# compute normalized adj_matrix
		adj_matrix = degree_reverse @ adj_matrix
		
		# compute new feature
		new_feature = self.w(adj_matrix @ all_feature) # (n_neighbor + n_target, dim_z)

		# get targets' feature
		target_feature = new_feature[n_neighbor:, :]

		# output layer
		output = self.output_layer(target_feature)

		return output