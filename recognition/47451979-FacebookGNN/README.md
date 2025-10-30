# Implementing a Graph Neural Network to Classify Facebook Pages

## Dependencies

- bokeh 3.8.0: Required for hammer-bundle plot (umap/umap.plot).
- dask 2025.10.0: Required for hammer-bundle plot (umap/umap.plot).
- datashader 0.18.2: Required for hammer-bundle plot (umap/umap.plot).
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
| *UMAP embedding with true labels denoted by colour.* |

Disregarding the classes and using 15-nearest neighbours to connect the nodes gives:

| ![Connectivity graph.](/recognition/47451979-FacebookGNN/assets/HammerAll.png) |
|:--:|
| *This representation is used to emphasize the spread of the data.* |

The features of each page are then expressed in a json file.

## Algorithm

The algorithm works by classifying each page based on its features and edges. Graph neural networks are heavily based on convolutional neural networks, however less impacted by larger data. A good model for the facebook data would have many uses such as if a new batch of pages that belong to a fifth category needed to be sorted into the four already existing categories. The basic architecture of the graph neural net was provided by Kaggle, additionally the pre-processing was done with code from Kaggle.

The algorithm was enhanced from the Kaggle version by adding layers, increasing the number of epochs, and partitioning the test data into separate test and validation sets as the original code lacked a validation set. Furthermore, the Kaggle implementation was enhanced by moving the tensors to the GPU, thus allowing for more extensive testing to be done, uninhibited by runtime.

The split for the train, test, and validation sets was 80%, 10%, and 10% respectively. This was deemed reasonable as there was a significant decrease in accuracy when the training set was less than 70% of the data. Plotting the training and validation losses gives the following plot:

| ![UMAP embedding.](/recognition/47451979-FacebookGNN/assets/Losses.png) |
|:--:|
| *Training and validation losses across 2000 epochs.* |


## Results

This a UMAP embedding of the test data with true labels denoted by colour.

| ![Other graph.](/recognition/47451979-FacebookGNN/assets/TrueLabels.png) |
|:--:|
| *UMAP embedding with true labels.* |

The following graph is the UMAP embedding with the predicted labels denoted by colour.

| ![Another graph.](/recognition/47451979-FacebookGNN/assets/PredictedLabels.png) |
|:--:|
| *UMAP embedding with predicted lbales.* |

As can be seen these two graphs are fairly similar, the accuracy of this test was 87%.


