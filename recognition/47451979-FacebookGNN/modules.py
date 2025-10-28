import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from dataset import data

class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42) # I'm not too sure how this line works, I'll play around with it.
        self.conv1 = GCNConv(data.num_features, 64) # I want to try some different values here.
        self.conv2 = GCNConv(64, 32)

        self.conv3 = GCNConv(32, 16)
        self.conv4 = GCNConv(16, 8) # I want to try some different values here.
        self.conv5 = GCNConv(8, 4) # I want to try some different values here.
        self.conv6 = GCNConv(4, torch.unique(data.y).size(0)) # I want to try some different values here.
    
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
        return F.log_softmax(x, dim=1) # Fiddle with this.
