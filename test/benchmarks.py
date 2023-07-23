import sys
import os
# import pybind11
# Append the parent directory of racplusplus package to system path
# sys.path.append(os.path.join(os.path.abspath(__file__), "..", "..", "build"))
# sys.path.append("/Users/porterhunley/repos/racplusplus/build")
import numpy as np
import time
import racplusplus
import scipy as sp
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# print("\nSee how it performs with a connectivity matrix")

# Set up matrix by size and density

def run_test(rows, cols, seed):
    np.random.seed(seed)
    dense_matrix = np.random.random((rows, cols))

    max_merge_distance = .24
    batch_size = 1000
    no_processors = 1

    from sklearn.neighbors import kneighbors_graph
    knn_graph = kneighbors_graph(dense_matrix, 3, mode='connectivity', include_self=True)
    symmetric = knn_graph + knn_graph.T

    labels = racplusplus.rac(
        dense_matrix,
        max_merge_distance,
        symmetric, 
        batch_size,
        no_processors,
        "cosine")
    

# result = True
# seed = 82
# try:
#     while result:
#         print(f"Running test on seed {seed}")
#         result = run_test(10, 10, seed)
#         # if (seed == 29):
#         #     result = True
#         seed += 1
# except:
#     print(f"Test failed on seed {seed-1}.")
# print(f"Test failed on seed {seed-1}.")


# while True:
print(run_test(30, 30, 42))
print(run_test(40, 40, 42))
print(run_test(50, 50, 42))
print(run_test(60, 60, 42))
# print(run_test(20, 20, 83))
