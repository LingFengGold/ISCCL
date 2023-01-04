# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:39:23 2022

@author: uros
"""
from typing import List, Tuple, Union, Mapping, Any
from scipy.sparse.csr import csr_matrix
import numpy as np

from umap import UMAP

def reduce_dimensionality(embeddings: Union[np.ndarray, csr_matrix],
                           y: Union[List[int], np.ndarray] = None) -> np.ndarray:
    """ Reduce dimensionality of embeddings using UMAP and train a UMAP model
    Arguments:
        embeddings: The extracted embeddings using the sentence transformer module.
        y: The target class for (semi)-supervised dimensionality reduction
    Returns:
        umap_embeddings: The reduced embeddings
    """
    if isinstance(embeddings, csr_matrix):
        umap_model = UMAP(n_neighbors=15,
                               n_components=5,
                               metric='hellinger',
                               low_memory=self.low_memory).fit(embeddings, y=y)
    else:
        umap_model = UMAP(n_neighbors=8,
                            n_components=32,
                            min_dist=0.0,
                            metric='cosine',
                            low_memory=False)
        
        
        umap_model.fit(embeddings, y=y)
    umap_embeddings = umap_model.transform(embeddings)
    return np.nan_to_num(umap_embeddings)



# DEFAULT 5 n_components n_neighbors=15