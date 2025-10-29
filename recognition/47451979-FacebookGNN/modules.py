import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from dataset import data

# Using standard structure for graph neural networks,
# this is based on the code from Kaggle but with one
# extra layer and a different random seed.
class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42) # Random seed.
        self.conv1 = GCNConv(data.num_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 32)
        self.conv4 = GCNConv(32, 16)
        self.conv5 = GCNConv(16, 8)
        self.conv6 = GCNConv(8, torch.unique(data.y).size(0))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.conv6(x, edge_index)
        return F.log_softmax(x, dim=1)
