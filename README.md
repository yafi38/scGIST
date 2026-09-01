# scGIST

[![CI](https://github.com/yafi38/scGIST/actions/workflows/ci.yml/badge.svg)](https://github.com/yafi38/scGIST/actions/workflows/ci.yml)

scGIST is a deep neural network that designs sc-ST gene panel through constrained feature selection. Additionally, scGIST enables genes of interest to be prioritized for panel inclusion while still adhering to its size restriction.

## Installation

scGIST isn't published on PyPI yet — install it from source with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/yafi38/scGIST.git
cd scGIST
uv sync
```

This installs scGIST into a local `.venv`; run scripts with `uv run python ...` or activate the
virtualenv directly. Requires Python 3.10–3.13.

## Usage

### Initialize the model

```python
from scgist import scGIST

gist = scGIST()
```

### Create the model

* Gene panel design to distinguish among cell types
  * `n_features`: number of genes/features
  * `n_classes`: number of classes/clusters/labels
  * `panel_size`: number of genes to be taken

```python
gist.create_model(n_features, n_classes, panel_size=panel_size)
```

* Including genes of interest and/or complexes of interest
  * `priority_scores`: priority scores of the genes of interest (see [Prioritize genes of interest](#prioritize-genes-of-interest))
  * `pairs`: list of complexes of interest

```python
gist.create_model(n_features, n_classes, panel_size=panel_size, priority_scores=priority_scores, pairs=pairs)
```

* Rigorously selecting the number of genes in the final panel as per `panel_size`
  * `strict`: when `True`, the model will select exactly the same amount of genes specified by `panel_size`; when `False`, the model will select less than or equal to that number

```python
gist.create_model(n_features, n_classes, panel_size=panel_size, strict=False)
```

### Compile the model

```python
gist.compile_model()
```

### Train the model

* Training the scGIST model, which requires either an `AnnData` object and the name of the column holding cell type labels, or raw `X`/`y` arrays
  * `adata`: annotated data matrix
  * `label_column`: `adata.obs` column name that contains the cell type labels
  * `epochs`: number of epochs

```python
gist.train_model(adata, label_column, epochs=200)
```

### Get the marker names (gene panel)

* `plot_weights`: when `True`, the weights of the genes in the panel are plotted in a bar chart

```python
markers = gist.get_markers_names(adata, plot_weights=True)
```

### Get accuracy and F1 score with a classifier

* Test performance of the gene panel with a classifier
  * `adata`: annotated data matrix
  * `label_column`: `adata.obs` column name that contains the cell type labels
  * `markers`: indices of the selected gene panel (`gist.get_markers_indices()`)
  * `clf`: a classifier (if `None`, defaults to KNN)

```python
from scgist import test_classifier

markers_indices = gist.get_markers_indices()
accuracy, f1 = test_classifier(adata, label_column, markers=markers_indices)
```

### Prioritize genes of interest

* Prioritize genes of interest to increase their probability of being included in the gene panel
  * Read a CSV file that contains gene names and their priority. It must contain columns named `gene_name` and `priority`
  * Convert it to a priority score list before creating the model

```python
import pandas as pd
from scgist import get_priority_score_list

gene_priorities = pd.read_csv(path_to_csv_file)
priority_scores = get_priority_score_list(adata, gene_priorities)

gist.create_model(n_genes, n_classes, panel_size=panel_size, priority_scores=priority_scores, alpha=0.2, beta=0.5)
```

## Examples

* [How to preprocess a dataset](examples/preprocess.ipynb)
* [How to run scGIST end-to-end](examples/run_scgist.ipynb)

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run mypy src
```

## Citation

This repository is under active development. To reproduce the results from
the paper (Genome Biology, 2024), use the code version archived at
publication time: <https://doi.org/10.5281/zenodo.10467039>.
