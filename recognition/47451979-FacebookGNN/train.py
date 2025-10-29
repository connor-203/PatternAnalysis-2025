from dataset import data, train_empty, test_empty, device
from modules import GNN
import matplotlib.pyplot as plt
import torch


# Note, umap.plot requires pandas matplotlib datashader
# bokeh holoviews scikit-image and colorcet to be installed

# Whether you want to see graphs at the end.
plotting = True

# This bit of code is fairly closely based on the code from medium, I
# have suspicions though that medium got it from Kaggle. In any case,
# this exact block of code is used for quite a few machine learning problems.
model = GNN().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

times = []
losses_train = []
losses_val = []

for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index)
    loss = criterion(pred[train_empty], data.y[train_empty])
    loss.backward()
    optimizer.step()

    times.append(epoch)
    losses_train.append(loss.item())

    # Calculate validation loss. Code loosely based on something found on stackoverflow.
    model.eval()
    pred_val = model(data.x, data.edge_index)
    loss_val = criterion(pred_val[test_empty], data.y[test_empty])
    losses_val.append(loss_val.item())

    if (epoch+1) % 100 == 0:
        print(f"Epoch: {epoch + 1:03d}, Training Loss: {loss:.4f}, Validation Loss: {loss_val:.4f}")

plt.plot(times, losses_train, label="Training Loss")
plt.plot(times, losses_val, label="Validation Loss")
plt.title("Training and Validation Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("assets/Training and Validation Losses for a GNN.png")
