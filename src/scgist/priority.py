import numpy as np


def get_priority_score_list(adata, gene_priorities):
    if 'gene_name' not in gene_priorities.columns or 'priority' not in gene_priorities.columns:
        print('priority_scores must contain "gene_name" and "priority" column')
        return None

    n_genes = adata.X.shape[1]
    priority_scores = np.zeros(n_genes)

    for _, row in gene_priorities.iterrows():
        if row['gene_name'] in adata.var_names:
            ind = adata.var_names.get_loc(row['gene_name'])
            priority_scores[ind] = row['priority']

    return priority_scores.tolist()
