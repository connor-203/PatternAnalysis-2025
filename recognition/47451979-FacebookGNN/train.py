from dataset import data, train_empty, test_empty
from modules import GNN
import torch

# This bit of code is fairly closely based on the code from medium, I
# have suspicions though that medium got it from Kaggle. In any case,
# this exact block of code is used for quite a few machine learning problems.
model = GNN()
error = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index)
    loss = error(pred[train_empty], data.y[train_empty])
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch: {epoch + 1:03d}, Loss: {loss:.4f}")
        
# This little block of code is also based off the code from medium.com. Any port in storm I suppose.
model.eval()
result = model(data.x, data.edge_index) # May be able to get rid of the equals bit, just do model(xyz)
pred = result.argmax(dim=1)
test_correct = pred[test_empty] == data.y[test_empty]
test_acc = int(test_correct.sum()) / int(test_empty.sum())
print(test_acc)

# First test had an accuracy of 56%, you started somewhere Connor.
    
