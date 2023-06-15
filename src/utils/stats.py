import numpy as np

# uses Euclidean distance
def within_cluster_variance(cluster):
    cluster = np.array(cluster)
    size = len(cluster)
    sum_of_squares = 0

    for i in range(size-1):
        vector1 = cluster[i]
        subcluster = cluster[i+1:]
        for vector2 in subcluster:
            diffs = vector1 - vector2
            sum_of_squares += sum(np.square(diffs))
    
    return sum_of_squares/size