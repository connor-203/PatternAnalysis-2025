from dataset import data, train_empty, test_empty, device
from modules import GNN
import matplotlib as plt
import numpy as np
import torch
import umap
import umap.plot

# Note, umap.plot requires pandas matplotlib datashader
# bokeh holoviews scikit-image and colorcet to be installed

# This bit of code is fairly closely based on the code from medium, I
# have suspicions though that medium got it from Kaggle. In any case,
# this exact block of code is used for quite a few machine learning problems.
plotting = True
model = GNN().to(device)
error = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

times = []
losses = []

model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index)
    loss = error(pred[train_empty], data.y[train_empty])
    loss.backward()
    optimizer.step()

    times.append(epoch)
    losses.append(loss)

    if (epoch+1) % 100 == 0:
        print(f"Epoch: {epoch + 1:03d}, Loss: {loss:.4f}")
        
# This little block of code is also based off the code from medium.com. Any port in storm I suppose.
model.eval()
result = model(data.x, data.edge_index) # May be able to get rid of the equals bit, just do model(xyz)
pred = result.argmax(dim=1) # I don't know what this does yet.
test_correct = pred[test_empty] == data.y[test_empty]
test_acc = int(test_correct.sum()) / int(test_empty.sum())
print(test_acc)

finish_model = model
# First test had an accuracy of 56%, you started somewhere Connor.


# print(pred)
# print(pred.shape)
# print(test_empty)
# print(test_empty.shape)

new_label = []
for i in data.y:
    if i == 0:
        new_label.append("Company")
    elif i == 1:
        new_label.append("Government")
    elif i == 2:
        new_label.append("Politician")
    elif i == 3:
        new_label.append("TV Show")

    
# print(data.y)
# print(new_label)
if plotting:
    # UMAP doesn't have GPU support so the data must be transferred to the CPU.
    mapper = umap.UMAP().fit(data.x.to('cpu'))
    plotted = umap.plot.points(mapper, labels=np.array(new_label))
    # connect = umap.plot.connectivity(mapper, show_points=True)

    # Hammer_bundle operation requires dask and scikit-image
    connect = umap.plot.connectivity(mapper, edge_bundling='hammer')
    umap.plot.show(plotted)

print("Success")
