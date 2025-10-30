import os
import pandas as pd
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
import numpy as np
import warnings

# This line serves one purpose, to omit the warnings
# from the import statement from umap.plot. It's a
# little heavy-handed of a fix, so it does also
# suppress all other warnings.
warnings.filterwarnings("ignore")

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# The data used was saved locally.
data_dir = "facebook_large"

# Portion of data used for training, validation and testing.
trainPercent = 0.8
validPercent = 0.1
testPercent = 0.1

def load_node_csv(path, index_col, **kwargs):
    df = pd.read_csv(path, **kwargs)
    mapping = {i: node_id for i, node_id in enumerate(df[index_col].unique())}

    # Load node features
    with open(os.path.join(data_dir, "musae_facebook_features.json"), "r") as json_file:
        features_data = json.load(json_file)

    xs = []
    for index, node_id in mapping.items():
        features = features_data.get(str(index), [])
        if features:
            # Create tensor from feature vector
            features_tensor = torch.tensor(features, dtype=torch.float).to(device)
            xs.append(features_tensor)
        else:
            xs.append(torch.zeros(1, dtype=torch.float).to(device))
    
    # Pad features to have vectors of the same size
    padded_features = pad_sequence([seq.detach().clone() for seq in xs], batch_first=True, padding_value=0)
    mask = padded_features != 0 # mask to indicate which features were padded
  
    # Create tensor of normaized features for nodes
    mean = torch.mean(padded_features[mask].float()).to(device)
    std = torch.std(padded_features[mask].float()).to(device)

    x = (padded_features - mean) / (std + 1e-8)  # final x tensor with normalized features

    return x

# Contains the PyTorch tensors containing feature vectors of the 22,470 Facebook pages.
x = load_node_csv(path=os.path.join(data_dir, "musae_facebook_target.csv"), index_col="facebook_id")

def load_labels_csv(path, label_col, **kwargs):
    df = pd.read_csv(path, **kwargs)

    # Convert class_labels to numeric data types in order to create tensors
    label_categories = df[label_col].astype("category").cat.categories

    df["class_label_code"] = pd.Categorical(df[label_col], categories=label_categories).codes

    y = torch.tensor(df["class_label_code"].values, dtype=torch.long).to(device)

    return y

# PyTorch tensor for the 22,470 Facebook pages containing the page_type label in the dataset.
y = load_labels_csv(path=os.path.join(data_dir, "musae_facebook_target.csv"), label_col="page_type")

def load_edge_csv(path, src_index_col, dst_index_col, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = df[src_index_col].values
    dst = df[dst_index_col].values
    
    # Converting from a list to a tensor is slow,
    # this converts to a np.array first.
    srcdist = np.array([src, dst])
    
    edge_index = torch.tensor(srcdist).to(device)

    return edge_index

edge_index = load_edge_csv(path=os.path.join(data_dir,
            "musae_facebook_edges.csv"), src_index_col="id_1", dst_index_col="id_2")

data = Data(x=x, edge_index=edge_index, y=y)

# Calculate how many nodes in the training and validation sets,
sliced = int(np.round(data.num_nodes * trainPercent))
valSlice = int(np.round(data.num_nodes * validPercent))

# Set up train, validation, and test masks.
train_empty=torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
train_empty[: sliced]=True

val_empty=torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
val_empty[sliced: sliced + valSlice] = True

test_empty=torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
test_empty[sliced + valSlice: ]=True

