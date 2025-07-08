import numpy as np
import math
from numba import njit, prange


@njit(parallel=True)
def iterate_fmin(distance_matrix, current, next_step, lr):
    size = len(projection)
    for i in prange(size):
        forcex=0
        forcey=0
        for j in prange(size):
            if(i==j):
                continue
            ox = current[j][0] - current[i][0]
            oy = current[j][1] - current[i][1]
            dr2 = math.sqrt(ox*ox+oy*oy)
            r = (i + j - math.fabs(i - j)) / 2  # min(i,j)
            s = (i + j + math.fabs(i - j)) / 2  # max(i,j)
            drn = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]
            forcex+=ox*((drn-dr2)/dr2)
            forcey+=oy*((drn-dr2)/dr2)
        next_step[i][0]=[current[i][0]-forcex*lr/size]
        next_step[i][1]=[current[i][1]-forcey*lr/size]


def execute_fmin(distance_matrix, projection, max_it, learning_rate0=0.5, lrmin= 0.05, decay=0.95):
    size = len(projection)
    for i in prange(max_it):
        learning_rate = max(learning_rate0 * math.pow((1 - k / max_it), decay), lrmin)
        iterate_fmin(distance_matrix, np.copy(projection), projection, lr)
        
    # setting the min to (0,0)
    min_x = min(projection[:, 0])
    min_y = min(projection[:, 1])
    for i in range(size):
        projection[i][0] -= min_x
        projection[i][1] -= min_y
