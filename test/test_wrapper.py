import sys
import os
import pybind11
# Append the parent directory of racplusplus package to system path
sys.path.append(os.path.join(os.path.abspath(__file__), "..", "..", "build"))
sys.path.append("/Users/danielfrees/Desktop/racplusplus/build")
print(f"Sys Path: {sys.path}")
print(f"Python Version: {sys.version}")
import racplusplus
import numpy as np
import scipy.sparse as sp

print(f"Found racplusplus: {racplusplus}")

print("\n---------------------First Python racplusplus package test (Basic IO):---------------------")
racplusplus.simple_pybind_io_test()
print("---------------------Basic IO test complete.---------------------")

print("\n---------------------Second Python racplusplus package test (Fully wrapped RAC):---------------------")
racplusplus.test_rac()
print("---------------------Fully wrapped RAC test complete.---------------------")


#--------------------------Test Actual RAC Endpoint------------------------------
print("\n---------------------Third Python racplusplus package test (Numpy & Scipy <> RAC Interface):---------------------")

# Set the random seed
np.random.seed(10)

# Generate a random matrix of size 10,000 x 768
test_matrix = np.random.random((10000, 768))

#------------------------- Generate sparse unweighted connectivity matrix-------------------------
print("\nGenerating sparse unweighted connectivity matrix (This takes around 20 seconds for size 10k x 10k)...")
# Set up matrix by size and density
rows = 10000
cols = 10000
density = 0.01

# Number of ones and zeros
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
symmetric_connectivity_matrix = sp.csc_matrix((data, (all_indices, all_cols)), shape=(rows, cols))
print("Done generating sparse symmetric unweighted connectivity matrix.\n")

#--------------------------Set up hyperparameters----------------------------------
max_merge_distance = .035
batch_size = 1000
no_processors = 0

print("Running RAC from Python using numpy data matrix and scipy sparse csc connectivity matrix.")
labels = racplusplus.rac(test_matrix, max_merge_distance, symmetric_connectivity_matrix, "symmetric", batch_size, no_processors)
print(f"Point Cluster Assignments: {labels}")

print("---------------------Numpy & Scipy <> RAC Interface test complete.---------------------")


#Generate empty sparse matrix
print("\n---------------------Fourth Python racplusplus package test (Numpy & Scipy <> RAC Interface w/o Conn Matrix):---------------------")
connectivity_matrix = sp.lil_matrix((0, 0))

print("Running RAC from Python using numpy data matrix and empty scipy sparse lil connectivity matrix.")
labels = racplusplus.rac(test_matrix, max_merge_distance, connectivity_matrix, "full", batch_size, no_processors)
print(f"Point Cluster Assignments: {labels}")

print("---------------------Numpy & Scipy <> RAC Interface w/o Conn Matrix test complete.---------------------")
#--------------------------End Test Actual RAC Endpoint------------------------------


print("\nPython Script racplusplus package test complete!\n")