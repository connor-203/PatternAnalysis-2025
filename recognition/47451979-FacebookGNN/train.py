from dataset import data, train_empty, test_empty, val_empty, device
from modules import GNN
import matplotlib.pyplot as plt
import torch


# Note, umap.plot requires pandas matplotlib datashader
# bokeh holoviews scikit-image and colorcet to be installed

# Whether you want to see graphs at the end.
plotting = True

# Setting up model, loss function, and optimiser.
model = GNN().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

times = []
losses_train = []
losses_val = []

for epoch in range(2000):
    # Training block.
    model.train()
    optimizer.zero_grad() # Zero the gradients.
    pred = model(data.x, data.edge_index) # Predict on all nodes.
    loss = criterion(pred[train_empty], data.y[train_empty]) # Look only at train nodes for error.
    loss.backward()
    optimizer.step()

    # Append epoch number and training loss.
    times.append(epoch)
    losses_train.append(loss.item())

    # Validation block.
    model.eval()
    pred_val = model(data.x, data.edge_index) # Predict on all nodes.
    loss_val = criterion(pred_val[val_empty], data.y[val_empty]) # Look only at validation nodes.

    # Append validation loss.
    losses_val.append(loss_val.item())

    # Print training and validation loss.
    if (epoch+1) % 100 == 0:
        print(f"Epoch: {epoch + 1:03d}, Training Loss: {loss:.4f}, Validation Loss: {loss_val:.4f}")

# Plot training and validation loss.
plt.plot(times, losses_train, label="Training Loss")
plt.plot(times, losses_val, label="Validation Loss")
plt.title("Training and Validation Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Training and Validation Losses for a GNN.png")


