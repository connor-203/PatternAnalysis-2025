import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from dataset import data_train

class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(123) # I'm not too sure how this line works, I'll play around with it.
        self.conv1 = GCNConv(data_train.num_features, 32) # I want to try some different values here.
        self.conv2 = GCNConv(32, torch.unique(data_train.y).size(0))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1) # Fiddle with this.
