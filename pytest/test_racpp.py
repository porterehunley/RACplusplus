import pytest
import scipy as sp
import numpy as np
from sklearn.cluster import AgglomerativeClustering

import racplusplus

SEED = 42

@pytest.fixture(scope="module")
def sparse_matrix():
    np.random.seed(42)

    density =.1
    rows = 100
    cols = 100

    num_ones = int(rows * cols * density)

    one_indices = np.random.choice(rows * cols, num_ones, replace=False)

    rows_indices, cols_indices = np.unravel_index(one_indices, (rows, cols))
    all_indices = np.concatenate([rows_indices, cols_indices, cols_indices, rows_indices])
    all_indices = np.concatenate([all_indices, [rows-1]*rows, np.arange(rows)])

    all_cols = np.concatenate([cols_indices, rows_indices, rows_indices, cols_indices])
    all_cols = np.concatenate([all_cols, np.arange(rows), [rows-1]*rows])

    data = np.ones(len(all_indices), dtype=bool)

    symmetric_connectivity_matrix = sp.sparse.csc_matrix((data, (all_indices, all_cols)), shape=(rows, cols))

    return symmetric_connectivity_matrix


@pytest.fixture(scope="module")
def dense_matrix():
    np.random.seed(42)

    return np.random.random((100, 768))


def test_rac_sparse_cosine(dense_matrix, sparse_matrix):
    max_merge_distance = .24
    batch_size = 1000
    no_processors = 8

    labels = racplusplus.rac(
        dense_matrix,
        max_merge_distance,
        sparse_matrix, 
        batch_size,
        no_processors,
        "cosine")

    assert len(set(labels)) == 33


def test_dense_matrix_cosine(dense_matrix):
    max_merge_distance = .24
    batch_size = 1000
    no_processors = 8

    labels = racplusplus.rac(
        dense_matrix,
        max_merge_distance,
        None, 
        batch_size,
        no_processors,
        "cosine")
    
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        linkage='average',
        distance_threshold=max_merge_distance, 
        metric='cosine').fit(dense_matrix)
    
    assert len(set(labels)) == len(set(clustering.labels_))


def test_euclidean_pairwise():
    np.random.seed(42)  # for reproducibility
    X = np.random.rand(5, 3)

    from sklearn.metrics.pairwise import euclidean_distances
    sklearn_distances = euclidean_distances(X, X)

    rac_distances = racplusplus._pairwise_euclidean_distance(X, X)

    print (sklearn_distances)
    print (rac_distances)

    assert np.allclose(sklearn_distances, rac_distances)


def test_cosine_pairwise():
    np.random.seed(42)  # for reproducibility
    X = np.random.rand(5, 3)

    from sklearn.metrics.pairwise import cosine_distances 
    sklearn_distances = cosine_distances(X, X)

    rac_distances = racplusplus._pairwise_cosine_distance(X, X)

    assert np.allclose(sklearn_distances, rac_distances) 


def test_dense_matrix_euclidean(dense_matrix):
    max_merge_distance = 11.1
    batch_size = 1000
    no_processors = 8

    labels = racplusplus.rac(
        dense_matrix,
        max_merge_distance,
        None, 
        batch_size,
        no_processors,
        "euclidean")
    
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        linkage='average',
        distance_threshold=max_merge_distance, 
        metric='euclidean').fit(dense_matrix)
    
    assert len(set(labels)) == len(set(clustering.labels_))


def test_rac_sparse_euclidean(dense_matrix, sparse_matrix):
    max_merge_distance = 11.1
    batch_size = 1000
    no_processors = 8

    from sklearn.neighbors import kneighbors_graph
    knn_graph = kneighbors_graph(dense_matrix, 30, mode='connectivity', include_self=True)
    symmetric = knn_graph + knn_graph.T

    labels = racplusplus.rac(
        dense_matrix,
        max_merge_distance,
        symmetric, 
        batch_size,
        no_processors,
        "euclidean")

    assert len(set(labels)) == 7 
