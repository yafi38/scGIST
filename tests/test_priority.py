import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scgist.priority import get_priority_score_list


def _make_adata(gene_names: list[str]) -> ad.AnnData:
    n_genes = len(gene_names)
    return ad.AnnData(X=np.zeros((2, n_genes), dtype=np.float32), var=pd.DataFrame(index=gene_names))


def test_scores_are_assigned_at_the_correct_gene_index():
    adata = _make_adata(["GENE_A", "GENE_B", "GENE_C"])
    priorities = pd.DataFrame({"gene_name": ["GENE_C", "GENE_A"], "priority": [5.0, 2.0]})

    scores = get_priority_score_list(adata, priorities)

    assert scores == pytest.approx([2.0, 0.0, 5.0])


def test_genes_not_present_in_adata_are_silently_ignored():
    adata = _make_adata(["GENE_A", "GENE_B"])
    priorities = pd.DataFrame({"gene_name": ["UNKNOWN"], "priority": [9.0]})

    scores = get_priority_score_list(adata, priorities)

    assert scores == pytest.approx([0.0, 0.0])


def test_missing_required_columns_raises_value_error():
    adata = _make_adata(["GENE_A"])
    bad_priorities = pd.DataFrame({"name": ["GENE_A"], "priority": [1.0]})

    with pytest.raises(ValueError):
        get_priority_score_list(adata, bad_priorities)
