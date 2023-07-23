import pickle
import numpy as np
import scipy as sp

def create_sparse_matrix(rows, cols, density):
    # Create a random sparse matrix
    np.random.seed(0)

    num_ones = int(rows * cols * density)

    # Generate random indices for ones
    one_indices = np.random.choice(rows * cols, num_ones, replace=False)

    # Make sure both (i, j) and (j, i) indices exist
    rows_indices, cols_indices = np.unravel_index(one_indices, (rows, cols))
    all_indices = np.concatenate([rows_indices, cols_indices, cols_indices, rows_indices])
    all_indices = np.concatenate([all_indices, [rows-1]*rows, np.arange(rows)])

    all_cols = np.concatenate([cols_indices, rows_indices, rows_indices, cols_indices])

    # Include new cols indices for the connections from the last row to all other rows
    all_cols = np.concatenate([all_cols, np.arange(rows), [rows-1]*rows])

    # Create a boolean array for data
    data = np.ones(len(all_indices), dtype=bool)

    # Create the sparse symmetric matrix
    symmetric_connectivity_matrix = sp.sparse.csc_matrix((data, (all_indices, all_cols)), shape=(rows, cols))
    print("Done generating sparse unweighted connectivity matrix.\n")
    # pickle the matrix
    with open('connectivity_matrix.pkl', 'wb') as f:
        pickle.dump(symmetric_connectivity_matrix, f)

if __name__ == "__main__":
    create_sparse_matrix(20000, 20000, .3)