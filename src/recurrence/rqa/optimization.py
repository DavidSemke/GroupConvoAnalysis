import numpy as np
import matplotlib.pyplot as plt
from nolitsa import delay, dimension, noise

# Uses delayed mutual information to determine optimal delay
# The delay correpsonding to the first local minimum of the DMI 
# function is returned if first_min is true
# If not first_min or there is no local minimum, the delay that
# causes the DMI function to fall below threshold 1/e is returned
def optimal_delay(data_pts, max_delay, first_min=False, plot=False):
    # compute delayed mutual information
    lag_range = np.arange(max_delay)
    dmi = delay.dmi(data_pts, maxtau=max_delay)
    
    if first_min:
        # noise used to remove insignificant minima
        min_delays = local_min_indexes(noise.sma(dmi, hwin=1)) + 1

    if not first_min or np.size(min_delays) == 0:
        lower_threshold = 1/np.e

        for i in range(1, len(dmi)):
            
            if dmi[i] >= lower_threshold: continue
            
            min_delays = [i]
            break

        else:
            raise Exception('Value of max_delay is too small')

    if plot:
        plt.figure(1)

        plt.subplot(211)
        plt.title(r'Delay estimation')
        plt.ylabel(r'Delayed mutual information')
        plt.plot(lag_range, dmi, min_delays, dmi[min_delays], 'o')

        plt.figure(2)
        plt.subplot(121)
        plt.title(r'Time delay = %d' % min_delays[0])
        plt.xlabel(r'$x(t)$')
        plt.ylabel(r'$x(t + \tau)$')
        plt.plot(data_pts[:-min_delays[0]], data_pts[min_delays[0]:])

        plt.show()

    return min_delays[0]


def local_min_indexes(data):
    return (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1


# Returns a dim that minimizes the FNN (False Nearest Neighbors)
# percentage
# Consider increasing maxnum or decreasing window if all nearest
# neighbors of a point are of nonzero distance
def optimal_embed(
    data_pts, delay, max_dim, window=10, maxnum=None, plot=False
):
    
    if not maxnum:
        symbol_counts = {}

        for pt in data_pts:
            if not pt in symbol_counts:
                symbol_counts[pt] = 1
            else:
                symbol_counts[pt] += 1
            
        max_count = max(symbol_counts.values())

        # Use count of most frequent symbol to define maxnum
        # This ensures that at least one near neighbor will be of
        # nonzero distance
        maxnum = 2 * window + max_count + 1
    
    # this conversion to float64 is needed for nolitsa code to work
    data_pts = np.float64(data_pts)
    # create range of dimensions to consider
    dim = np.arange(1, max_dim + 1)

    f1, f2, f3 = dimension.fnn(
        data_pts, dim, delay, window=window, maxnum=maxnum, 
        parallel=False
    )

    optimal_dim = len(f3)

    for i in range(1, len(f3)):
        curr_fnn_perc = f3[i]
        prior_fnn_perc = f3[i-1]
        
        if prior_fnn_perc > curr_fnn_perc: continue
        
        optimal_dim = i
        break
    
    if plot:
        plt.xlabel(r'Embedding dimension $d$')
        plt.ylabel(r'FNN (%)')
        plt.plot(dim, 100 * f1, 'bo--', label=r'Test I')
        plt.plot(dim, 100 * f2, 'g^--', label=r'Test II')
        plt.plot(dim, 100 * f3, 'rs-', label=r'Test I or II')
        plt.legend()

        plt.show()

    # return the embedding dim that is followed by a dim
    # with a greater or equal FNN percentage
    return optimal_dim