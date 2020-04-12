import numpy as np
import torch
import torch.autograd as autograd
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
# 定义输入输出
class LSTMTagger(nn.Module):
	def __init__(self, node_size, input_dim, hidden_dim, out_dim, pre_embedding, n_layers = 1, dropout = 0.5):
		super(LSTMTagger, self).__init__()
		self.node_size = node_size
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.out_dim = out_dim
		self.pre_embedding = pre_embedding
		# 随机生成的
		self.embedding = nn.Embedding(node_size, input_dim)
		self.embedding.weight = nn.Parameter(pre_embedding)
		#把pre_embedding的值复制过去
		# self.lstm = nn.LSTM(input_dim, hidden_dim)
		self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
		self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout)

						 
	def forward(self, paths_between_one_pair_id):
# forward
		sum_hidden = Variable(torch.Tensor(), requires_grad=True)
		paths_size = len(paths_between_one_pair_id)

		for i in range(paths_size):
			path = paths_between_one_pair_id[i]
			path_size = len(path)
			path_embedding = self.embedding(path)
			path_embedding = path_embedding.view(path_size, 1, self.input_dim)
			if torch.cuda.is_available():
				path_embedding = path_embedding.cuda()
			_, h = self.lstm(path_embedding)
			if i == 0:
					sum_hidden = h[0]
			else:
					sum_hidden = torch.cat((sum_hidden, h[0]), 1)
		#torch.cat 连接sum_hidden和h[0] maxpool
		pool = nn.MaxPool2d((paths_size, 1), stride=(paths_size, 1))
		pool = nn.AvgPool2d
		max_pool = pool(sum_hidden)
		out = self.linear(max_pool)
		out = F.sigmoid(out)
		return out
	# 输出的是二维的值 1或0的百分比
