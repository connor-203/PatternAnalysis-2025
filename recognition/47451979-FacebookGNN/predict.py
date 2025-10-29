from dataset import data, test_empty
from train import model
import numpy as np
import matplotlib.pyplot as plt
import umap
import umap.plot


# This little block of code is also based off the code from medium.com. Any port in storm I suppose.
model.eval()
result = model(data.x, data.edge_index)
pred = result.argmax(dim=1) # Reduces to 1 dimension

test_correct = pred[test_empty] == data.y[test_empty]
test_acc = int(test_correct.sum()) / int(test_empty.sum())
print(f"This test had an accuracy of: {test_acc}")

# For plotting all nodes
xs = data.x
ys = data.y
ps = pred

# For plotting only the test nodes.
# xs = data.x[test_empty]
# ys = data.y[test_empty]
# ps = pred[test_empty]

new_label = []
for i in ys:
    if i == 0:
        new_label.append("Company")
    elif i == 1:
        new_label.append("Government")
    elif i == 2:
        new_label.append("Politician")
    elif i == 3:
        new_label.append("TV Show")

pred_label = []
for i in ps:
    if i == 0:
        pred_label.append("Company")
    elif i == 1:
        pred_label.append("Government")
    elif i == 2:
        pred_label.append("Politician")
    elif i == 3:
        pred_label.append("TV Show")


# UMAP doesn't have GPU support so the data must be transferred to the CPU.

mapperTest = umap.UMAP().fit(xs.to('cpu'))

plotted = umap.plot.points(mapperTest, labels=np.array(new_label))
plt.savefig("assets/True Labels.png")

plottedPred = umap.plot.points(mapperTest, labels=np.array(pred_label))
plt.savefig("assets/Predicted Labels.png")

# connect = umap.plot.connectivity(mapper, show_points=True)

# Hammer_bundle operation requires dask and scikit-image
connect = umap.plot.connectivity(mapperTest, edge_bundling='hammer')
plt.savefig("assets/Recent Hammer.png")

plt.show()
# umap.plot.show(plotted)

print("Success")
