from scGIST import scGIST
import scanpy as sc
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np

base_path = 'G:/My Programs/Thesis/scGIST'
data_path = base_path + '/data-2'
results_path = base_path + '/results/spatial_2'

adata = sc.read_h5ad(data_path + '/olfactory_intersect.h5ad')
adata.X = csr_matrix.todense(adata.X)
group_by = 'celltype'

n_genes = adata.X.shape[1]
n_classes = adata.obs[group_by].unique().size

panels = [25, 50, 75, 100, 125, 150, 175, 200]
# panels = [175, 200]
total_panels = len(panels)
alphas = np.zeros(total_panels)
genes_taken = np.zeros(total_panels)

counter = 0

for panel_size in panels:
    print('==========================================')
    print('running for panel:', panel_size)
    
    alpha_min = 0.001
    alpha_max = 2

    for i in range(10):
        alpha = (alpha_min + alpha_max) / 2
        gist = scGIST()
        gist.create_model(n_genes, n_classes, panel_size=panel_size, alpha=alpha)
        gist.compile_model()
        gist.train_model(adata, group_by, verbose=0, epochs=100)
        
        marker_count = gist.get_significant_marker_count()
        print(i, alpha, marker_count, panel_size)

        if marker_count > panel_size:
            alpha_min = alpha
        elif marker_count < panel_size:
            alpha_max = alpha
        else:
            break

    alphas[counter] = alpha
    genes_taken[counter] = marker_count
    counter = counter + 1

acc_data = pd.DataFrame({'panel_size': panels, 'genes_taken': genes_taken, 'alpha': alphas})
acc_data.to_csv(results_path + '/nm_sc_alphas.csv', index=False)