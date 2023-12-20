from data_paths import *
from scGIST import scGIST
import scanpy as sc
from scipy.sparse import csr_matrix
import pandas as pd

adata = sc.read_h5ad(SPATIAL_PATH_2 + '/olfactory_intersect.h5ad')
adata.X = csr_matrix.todense(adata.X)
# adata = sc.read_h5ad(SPATIAL_PATH_2 + '/seqfish_olfactory_final.h5ad')
group_by = 'celltype'

save_path = f'./results/spatial_2/'
n_genes = adata.X.shape[1]
n_classes = adata.obs[group_by].unique().size

panels = [25, 50, 75, 100, 125, 150]

for panel_size in panels:
    print('running for panel:', panel_size)
    gist = scGIST()
    gist.create_model(n_genes, n_classes, panel_size=panel_size, alpha=0.2)
    gist.compile_model()
    gist.train_model(adata, group_by, verbose=2, epochs=50)
    markers = gist.get_markers_names(adata, verbose=0, plot_weights=True)

    file_name = f'{save_path}intersect_{panel_size}_scrna_new.csv'
    pd.DataFrame(markers).to_csv(file_name, index=False, header=False)