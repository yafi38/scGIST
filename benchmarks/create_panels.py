import pandas as pd
import scanpy as sc
import numpy as np
from tqdm import tqdm

from scGIST import scGIST
from data_paths import *


def load_pbmc():
    return sc.read_h5ad(PBMC_PATH + '/pbmc3k.h5ad'), 'leiden'


def load_head_neck():
    return sc.read_h5ad(HEAD_NECK_PATH + '/head_neck.h5ad'), 'non-cancer cell type'


def load_endoderm():
    return sc.read_h5ad(ENDODERM_PATH + '/endoderm_post_process.h5ad'), 'CellType'


def load_tabula_sapiens():
    return sc.read_h5ad(TABULA_PATH + '/adata_train.h5ad'), 'celltype'


def load_data(dataset_name):
    if dataset_name == 'endoderm':
        return load_endoderm()
    elif dataset_name == 'pbmc':
        return load_pbmc()
    elif dataset_name == 'headneck':
        return load_head_neck()
    else:
        return load_tabula_sapiens()


if __name__ == '__main__':
    datasets = ['pbmc', 'headneck', 'tabula_sapiens', 'endoderm']

    for dataset in datasets:
        save_path = f'./results/{dataset}/nm'

        adata, group_by = load_data(dataset)

        n_genes = adata.X.shape[1]
        n_classes = adata.obs[group_by].unique().size

        for j in range(5):
            for i in tqdm(range(0, 18), unit='panel_sizes'):
                panel_size = 5 * i + 15

                gist = scGIST()
                gist.create_model(n_genes, n_classes, panel_size=panel_size, alpha=1.5)
                gist.compile_model()
                gist.train_model(adata, group_by, verbose=0, epochs=200)
                markers = gist.get_markers_names(adata, verbose=0, plot_weights=False)

                file_name = f'{save_path}{j}_{panel_size}.csv'
                pd.DataFrame(markers).to_csv(file_name, index=False, header=False)
