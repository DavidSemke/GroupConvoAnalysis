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


def variance(data_pts):
    total_data_pts = len(data_pts)
    mean = sum(data_pts)/total_data_pts
    return sum([(dp-mean)**2 for dp in data_pts])/total_data_pts


def median(data_pts):

    if len(data_pts) == 1: return data_pts[0]
    
    data_pts.sort()

    if len(data_pts) % 2 == 0:
        mid_right = len(data_pts) // 2
        mid_left = mid_right - 1
        median = round((mid_left + mid_right)/2, 1)
    else:
        median = len(data_pts) // 2
    
    return median