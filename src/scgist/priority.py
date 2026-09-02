from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from anndata import AnnData


def get_priority_score_list(adata: AnnData, gene_priorities: pd.DataFrame) -> list[float]:
    if 'gene_name' not in gene_priorities.columns or 'priority' not in gene_priorities.columns:
        raise ValueError('gene_priorities must contain "gene_name" and "priority" columns')

    assert adata.X is not None
    n_genes = adata.X.shape[1]
    priority_scores: npt.NDArray[np.float64] = np.zeros(n_genes)

    for _, row in gene_priorities.iterrows():
        if row['gene_name'] in adata.var_names:
            ind = adata.var_names.get_loc(row['gene_name'])
            priority_scores[ind] = row['priority']

    return priority_scores.tolist()
