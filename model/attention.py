import torch
from torch import nn
import torch.nn.functional as F
import numpy as np



class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        def forward(self,x):
            query = self.query(x)
            key = self.key(x)
            value = self.value(x)

            product = torch.matmul(query,key.transpose(-2,-1))
            scale = 1.0/(torch.sqrt(query.size(-1)))
            product = scale * product

            attention_weights = torch.nn.functional.softmax(product, dim =-1)

            weighted_sum = torch.matmul(attention_weights, value)

            return weighted_sum, attention_weights