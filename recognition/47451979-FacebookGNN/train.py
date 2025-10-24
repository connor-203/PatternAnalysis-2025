from dataset import data, data_train, train_empty
from modules import GNN
import torch

model = GNN()
error = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index)
    loss = error(pred[train_empty], data_train.y)
    loss.backward()
    optimizer.step()
    return loss

for epoch in range(1, 1001):
    loss = train()
    if epoch % 100 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
