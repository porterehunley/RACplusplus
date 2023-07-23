import time
from sklearn.cluster import AgglomerativeClustering
import numpy as np

MIN_DISTANCE=.15
np.random.seed(43)  # Set the seed for reproducibility

random_array = np.random.rand(10000, 768)
start = time.time()
clustering = AgglomerativeClustering(
    n_clusters=None, 
    linkage='average',
    distance_threshold=MIN_DISTANCE, 
    metric='cosine').fit(random_array)
  
end = time.time()
print(end - start)
