# Testing Github Markdown.

## Overview

## Dependencies

- bokeh 3.8.0: Required for hammer-bundle plot (umap/umap.plot).
- datashader 0.18.2: Required for hammer-bundle plot (umap/umap.plot).
- dask 2025.10.0: Required for hammer-bundle plot (umap/umap.plot).
- holoviews 1.21.0: Required for hammer-bundle plot (umap/umap.plot).
- matplotlib 3.9.2: Used for plotting.
- numpy 2.1.2: Used for arrays.
- Pandas 2.3.3: Used to read data.
- Python 3.12.10: Programming language used for all code.
- scikit-image 0.25.2: Required for hammer-bundle plot (umap/umap.plot)
- Torch 2.5.0+cu118: Used for creating and training the model.
- Torch-geometric 2.7.0: Used for creating graph neural network.
- umap 0.1.1: Used for plotting.
- umap-learn 0.5.9: Used for plotting.

## Project Structure

- `dataset.py`: Handles pre-processing of the data, uses code from Kaggle.
- `modules.py`: Creates the graph neural network using basic GNN structure.
- `train.py`: Trains and validates the GNN, also saves plot for training and validation losses.
- `predict.py`: Predicts classes of test nodes, creates plots of true and predicted labels.

## Data

The data is a representation of Facebook pages with links between pages being represented by node edges. Creating a UMAP embedding of the data gives the following plot:

| ![Connectivity graph.](/recognition/47451979-FacebookGNN/assets/NodeAll.png) |
|:--:|
| *A nother caption* |

Disregarding the classes and adding the node connections gives the following plot:

| ![Connectivity graph.](/recognition/47451979-FacebookGNN/assets/HammerAll.png) |
|:--:|
| *A nother caption* |

The features of each page are then expressed in a json file.

## Algorithm

The algorithm works by classifying each page based on its features and edges, a model that can do this with decent accuracy would have many uses such as if a new batch of pages that belong to a fifth category needed to be sorted into the four already existing categories. The basic architecture of the graph neural net was provided by Kaggle.

The algorithm was enhanced from the Kaggle version by adding layers, increasing the number of epochs, and partitioning the test data into separate test and validation sets as the original code lacked a validation set. Furthermore, the Kaggle implementation was enhanced by moving the tensors to the GPU, thus allowing for more extensive testing to be done.

The split for the train, test, and validation sets was 80%, 10%, and 10% respectively. This was deemed reasonable as there was a significant decrease in accuracy when the training set was less than 70% of the data.

| ![UMAP embedding.](/recognition/47451979-FacebookGNN/assets/Losses.png) |
|:--:|
| *A caption* |

Filler


## Results

88.4% Accuracy thank you very much.



| ![Other graph.](/recognition/47451979-FacebookGNN/assets/TrueLabels.png) |
|:--:|
| *A nother caption* |

to

| ![Another graph.](/recognition/47451979-FacebookGNN/assets/PredictedLabels.png) |
|:--:|
| *A nother caption* |

divide images.


| ![Another graph.](/recognition/47451979-FacebookGNN/assets/HammerTest.png) |
|:--:|
| *A nother caption* |

Data was Facebook kaggle data, preprocessing was done using code from Kaggle. This test is a draft. **Remind me to move the tensors to the GPU**.
