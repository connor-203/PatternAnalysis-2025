from dataset import data, test_empty
from train import model
import numpy as np
import matplotlib.pyplot as plt
import umap
import umap.plot


# This little block of code is based off code from medium.com. Any port in storm I suppose.
# Though it's based off the code from medium in the same way that the action of me driving around
# the corner is from my driving instructor.
model.eval()
result = model(data.x, data.edge_index)
pred = result.argmax(dim=1) # Reduces to 1 dimension

# Calculate accuracy with a list consisting of booleans representing whether
# pred[i] is the same as data.y[i].
test_correct = pred[test_empty] == data.y[test_empty]
test_acc = int(test_correct.sum()) / int(test_empty.sum())
print(f"This test had an accuracy of: {test_acc}")

# For plotting all nodes
# xs = data.x
# ys = data.y
# ps = pred

# For plotting only the test nodes.
xs = data.x[test_empty]
ys = data.y[test_empty]
ps = pred[test_empty]

# Somewhat clunky, but converts the category number back to the category.
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
mapper = umap.UMAP().fit(xs.to('cpu'))

# Plot true labels,
plotted = umap.plot.points(mapper, labels=np.array(new_label))
plt.savefig("assets/TrueLabels.png")

# Plot predicted labels.
plottedPred = umap.plot.points(mapper, labels=np.array(pred_label))
plt.savefig("assets/PredictedLabels.png")

# connect = umap.plot.connectivity(mapper, show_points=True)

# Hammer_bundle operation requires dask and scikit-image
connect = umap.plot.connectivity(mapper, edge_bundling='hammer')
plt.savefig("assets/HammerTest.png")

plt.show()
# umap.plot.show(plotted)

print("Success")


