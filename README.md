# scGIST

scGIST is a deep neural network that designs sc-ST gene panel through constrained feature selection. Additionally, scGIST enables genes of interest to be prioritized for panel inclusion while still adhering to its size restriction.

## Installation

Install the requirements using the following command (recommended python version is 3.7.11):
```
python setup.py
```

## Usage

### Initialize the model
```python
from scgist import scGIST
scGIST = scGIST()
```

### Create the model
* Gene panel design to distinguish among cell types 
    * n_features: number of cells
    * n_classes: number of classes/ clusters/ labels, 
    * panel_size: number of features to be taken
```python
scGIST.create_model(n_features, n_classes, panel_size)
```
* Including genes of interest and/or complexes of interest
    * weights: priotiy scores of the genes of interest
    * pairs: list of complexes of interest
```python
scGIST.create_model(n_features, n_classes, panel_size, weights, pairs)
```
* Rigorously selecting the number of genes in the final panel as per panel_size
    * strict: when True, the model will select exactly the same amount of genes specified by panel_size; when False, the model will select less than or equal to the number of genes specified by panel_size
```python
scGIST.create_model(n_features, n_classes, panel_size, weights, pairs, strict)
```

### Compile the model
```python
scGIST.compile_model()
```


### Train the model
* Training the scGIST model which requires the following inputs:
    * adata: annotated data matrix
    * epochs: number of epochs
```python
scGIST.train_model(adata, epochs)
```

### Get the markers names (gene panel)
* Plotting the gene panel with weights in a bar chart
    * plot_weights: when True, the weights of the genes in the panel will be plotted
```python
scGIST.get_markers_names(plot_weights)
```

### Get Accuracy and F1 score with a classifier
* Test performance of the gene panel with a classifier
    * adata: annotated data matrix
    * markers: indices of selected gene panel (scGIST.get_markers_indices())
    * labels: name of the cell types
    * clf: a classifier (if None, default is KNN)
```python
from scGIST import test_classifier
accuracy, f1_score = test_classifier(adata, markers, labels, clf)
```

### Prioritize genes of interest
* Prioritize genes of interest to increase their probability of being included in the gene panel
  * Read the csv file that contains gene names and their priority. The csv file must contain headers named `gene_name` and `priority`
  * Convert the dataframe to a python list using utility function before creating the model with the priority score list
```python
gene_priorities = pd.read_csv(path_to_csv_file)
priority_scores = get_priority_score_list(adata, gene_priorities)

gist.create_model(n_genes, n_classes, panel_size=panel_size, priority_scores=priority_scores, alpha=0.2, beta=0.5)
```












