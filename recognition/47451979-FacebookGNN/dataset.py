import os
import pandas as pd
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data


data_dir = "facebook_large"


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
            features_tensor = torch.tensor(features, dtype=torch.float)
            xs.append(features_tensor)
        else:
            xs.append(torch.zeros(1, dtype=torch.float))
    
    # Pad features to have vectors of the same size
    padded_features = pad_sequence([torch.tensor(seq) for seq in xs], batch_first=True, padding_value=0)
    mask = padded_features != 0 # mask to indicate which features were padded
  
    # Create tensor of normaized features for nodes
    mean = torch.mean(padded_features[mask].float())
    std = torch.std(padded_features[mask].float())

    x = (padded_features - mean) / (std + 1e-8)  # final x tensor with normalized features

    return x

x = load_node_csv(path=os.path.join(data_dir, "musae_facebook_target.csv"), index_col="facebook_id")
# x containing the PyTorch tensors containing feature vectors of the 22,470 Facebook pages.
print(x.shape)

def load_labels_csv(path, label_col, **kwargs):
    df = pd.read_csv(path, **kwargs)

    # Convert class_labels to numeric data types in order to create tensors
    label_categories = df[label_col].astype("category").cat.categories

    class_label_to_code = pd.DataFrame({
        "class_label": label_categories,
        "class_label_code": pd.Categorical(label_categories, categories=label_categories).codes
    })

    df["class_label_code"] = pd.Categorical(df[label_col], categories=label_categories).codes

    y = torch.tensor(df["class_label_code"].values, dtype=torch.long)

    return y

y = load_labels_csv(path=os.path.join(data_dir, "musae_facebook_target.csv"), label_col="page_type")
# a PyTorch tensor for the 22,470 Facebook pages containing the page_type label in the dataset.
print(y.shape)

def load_edge_csv(path, src_index_col, dst_index_col, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = df[src_index_col].values
    dst = df[dst_index_col].values
    edge_index = torch.tensor([src, dst])

    return edge_index

edge_index = load_edge_csv(path=os.path.join(data_dir, "musae_facebook_edges.csv"), src_index_col="id_1", dst_index_col="id_2")
# edge_index is a tensor of length two, one for the source node and the other for the target node
print(edge_index.shape)

data=Data(x=x, edge_index=edge_index, y=y)
print(data)

