# RAC++
🚧 This project is very much under development - use at your own risk! 🚧

RAC++ - Reciprocal Agglomerative Clustering (in C++). Performs a parallelized, bottom up Agglomerative clustering by pairing and merging reciprocal nearest neighbors. This allows RAC++ to scale to much larger datasets than traditional agglomerative clustering while keeping the results (almost always) the same.

All the positive attributes of Agglomerative clustering remain with RAC++ as well. It produces meaningful clusters with little parameter tuning and can create whole taxonomies as well. 

Based on the paper:
```
@article{DBLP:journals/corr/abs-2105-11653,
  author       = {Baris Sumengen and Anand Rajagopalan and Gui Citovsky and David Simcha and Olivier Bachem and Pradipta Mitra and Sam Blasiak and Mason Liang and Sanjiv Kumar},
  title        = {Scaling Hierarchical Agglomerative Clustering to Billion-sized Datasets},
  journal      = {CoRR},
  volume       = {abs/2105.11653},
  year         = {2021},
  url          = {https://arxiv.org/abs/2105.11653},
}
```

## Contributors

Porter Hunley — @porterehunley
Daniel Frees — @danielfrees

## How to use RAC++

RAC++ is available via PyPI for Python 3.8 and up on all major platforms. Install it via pip:
```
pip install racplusplus
```

The RAC++ API is very similar to traditional Agglomerative clustering:
```
import racplusplus

X = np.random.random((10000, dim))
labels = racplusplus.rac(
  X, max_merge_distance=.24, connectivity=None, batch_size=1000, no_processors=8, distance_metric="cosine"
)
```

It should be noted that only the ` Average ` linkage method is available as of writing this Readme.

- ` X ` the array of points
- ` max_merge_distance ` the merge threshold
- ` connectivity ` the optional connectivity matrix -- **must be symmetric**
- ` batch_size ` the batch sized used to calculate the distance matrix. Pick a number large enough for fast results but small enough to not overload your memory.
- ` no_processors ` the number of threads you want it to use, should be less than or equal to the number of cores available on your machine. 
- ` distance_metric ` either "cosine" or "euclidean"

As of right now, returning the whole tree is not yet available.

## Performance
RAC++ is designed to scale Agglomerative clustering to much larger datasets. It runs *significantly* faster than traditional Agglomerative clustering and scales better as well. Right now, RAC++ can run just fine on datasets in the hundreds of thousands, even in very high dimensions. We expect that to grow significantly as we add options to optimize towards a linear runtime.

Here are some benchmarking examples: [Benchmarking](https://github.com/mediboard/racplusplus/blob/main/notebooks/RACBenchmarks.ipynb)

**Results**
RAC++ produces the exact same results as Agglomerative clustering when the points are fully connected.

 Even if the connectivity is limited, the results are almost always the same or a *tad* off. However, there are some outlier cases when the results can differ wildly with limited connectivity, so it's a good idea to check the results visually with subsample of data.

## Development status
We're aiming to recreate as many features from traditional agglomerative clustering as is feasible for the RAC algorithm. 


|         Feature Name     |  Status |
|--------------------------|---------|
| Average Linkage          |   ✅     |
| Cosine distance          |   ✅     |
| Euclidean distance       |   ✅     |
| Ward Linkage             |   🚧     |
| Complete Linkage         |   🚧     |
| External distance matrix |   ❌     |
| Single Linkage           |   ❌     |
| Returning dendrogram     |   ❌     |
| Pre-set cluster input    |   ❌     |
