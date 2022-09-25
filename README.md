# scGIST

scGIST is a deep neural network that designs sc-ST gene panel through constrained feature selection. Additionally, scGIST enables genes of interest to be prioritized for panel inclusion while still adhering to its size restriction.

## Installation

Install the requirements using the following command:
```
python setup.py
```

## Usage

### Initialize the model
```
scGIST = scGIST()
```

### Create the model
* Gene panel design to distinguish among cell types 
    * n_features: number of cells
    * n_classes: number of classes/ clusters/ labels, 
    * panel_size: number of features to be taken
```
scGIST.create_model(n_features, n_classes, panel_size)
```
* Including genes of interest and/or complexes of interest
    * weights: priotiy scores of the genes of interest
    * pairs: list of complexes of interest
```
scGIST.create_model(n_features, n_classes, panel_size, weights, pairs)
```
* Rigorously selecting the number of genes in the final panel as per panel_size
    * strict: when True, the model will select exactly the same amount of genes specified by panel_size; when False, the model will select less than or equal to the number of genes specified by panel_size
```
scGIST.create_model(n_features, n_classes, panel_size, weights, pairs, strict)
```

### Compile the model
```
scGIST.compile_model()
```


### Train the model
* Training the scGIST model which requires the following inputs:
    * X: gene expression matrix
    * y: cell type labels
    * epochs: number of epochs
```
scGIST.train_model(X, y, epochs)
```

### Get the markers names (gene panel)
* Plotting the gene panel with weights in a bar chart
    * plot_weights: when True, the weights of the genes in the panel will be plotted
```
scGIST.get_markers_names(plot_weights)
```

### Get Accuracy and F1 score with a classifier
* Test performance of the gene panel with a classifier
    * X: gene expression matrix
    * y: cell type labels
    * markers: selected gene panel
    * labels: name of the cell types
    * clf: a classifier (if None, default is KNN)
```
scGIST.get_accuracy(X, y, markers, labels, clf)
```











