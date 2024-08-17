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