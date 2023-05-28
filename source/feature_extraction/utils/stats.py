# uses Euclidean distance
def within_cluster_variance(cluster):
    size = len(cluster)
    sum_of_squares = 0
    for i in range(len(cluster)-1):
        vector1 = cluster[i]
        subcluster = cluster[i+1:]
        for vector2 in subcluster:
            for j in range(len(vector2)):
                sum_of_squares += (vector1[j]-vector2[j])**2
    
    return sum_of_squares/size


def variance(data_points):
    total_data_points = len(data_points)
    mean = sum(data_points)/total_data_points
    return sum([(dp-mean)**2 for dp in data_points])/total_data_points