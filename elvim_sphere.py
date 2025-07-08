import numpy as np
import math
from numba import njit, prange


def from_2d(point):
    s = math.sin(point[1])
    return [math.cos(point[0])*s, math.sin(point[0])*s, math.cos(point[1])]

def to_2d(point):
    magnitude = math.sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2])
    clamped = math.max(-1, math.min(1, point[2]/magnitude))
    return [math.atan2(point[1], point[0]), math.acos(clamped)]

def slerp(A, B, t, omega):
    d = math.sin(omega)
    s0 = math.sin((1-t)*omega)
    s1 = math.sin(t*omega)
    return [
            (A[0]*s0 + B[0]*s1)/d
            (A[1]*s0 + B[1]*s1)/d
            (A[2]*s0 + B[2]*s1)/d
        ]

def slerp(A, B, t):
    dotA=A[0]*A[0]+A[1]*A[1]+A[2]*A[2]
    dotB=B[0]*B[0]+B[1]*B[1]+B[2]*B[2]
    dotAB=A[0]*B[0]+A[1]*B[1]+A[2]*B[2]
    math.acos(dotAB/(math.sqrt(dotA)*math.sqrt(dotB)))
    return slerp_unit_vector(A,B,t,omega)

@njit(parallel=True, fastmath=False)
def move_sph(ins1, distance_matrix, projection, learning_rate, scale):
    size = len(projection)
    total = len(distance_matrix)
    error = 0

    pointA = projection[ins1]

    for ins2 in prange(size):
        if ins1 == ins2:
            continue
        pointB = projection[ins2]

        x1x2 = pointB[0] - pointA[0]
        y1y2 = pointB[1] - pointA[1]

        dr2 = max(math.acos(pointA[0]*pointB[0] + pointA[1]*pointB[1] + pointA[2]*pointB[2]), 0.01)

        # getting te index in the distance matrix and getting the value
        r = (ins1 + ins2 - math.fabs(ins1 - ins2)) / 2  # min(ins1,ins2)
        s = (ins1 + ins2 + math.fabs(ins1 - ins2)) / 2  # max(ins1,ins2)
        drn = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))] *scale

        # calculate the movement
        delta = (drn - dr2) #* math.fabs(drn - dr2)

        res = slerp(pointA,pointB,exp(lr*delta/dr2),dr2)
        # moving
        projection[ins2]=res/math.sqrt(res[0]*res[0]+res[1]*res[1]+res[2]*res[2])


def iteration_sph(index, distance_matrix, projection, learning_rate, scale):
    size = len(projection)

    for i in range(size):
        ins1 = index[i]
        move_sph(ins1, distance_matrix, projection, learning_rate)


def execute_sph(distance_matrix, projection, max_it, learning_rate0=0.5, lrmin= 0.05, decay=0.95, scale):
    size = len(projection)
    # create random index
    index = np.random.permutation(size)
    
    # convert from 2D coordinates to 3D
    three_d = [from_2d(it) for it in projection]

    for k in range(max_it):
        learning_rate = max(learning_rate0 * math.pow((1 - k / max_it), decay), lrmin)
        iteration_sph(index, distance_matrix, projection, learning_rate, scale)

    return [to_2d(it) for it in three_d]





@njit(parallel=True)
def iterate_fmin(distance_matrix, current, next_step, lr, scale):
    size = len(projection)
    for i in prange(size):
        forcex=0
        forcey=0
        pointA = current[i]
        for j in prange(size):
            if(i==j):
                continue
            pointB = projection[ins2]

            ox = pointB[0] - pointA[0]
            oy = pointB[1] - pointA[1]
            pdot = ox*pointA[0]+oy*pointA[1]
            ox-=pointA[0]*pdot
            oy-=pointA[1]*pdot

            dr2 = math.sqrt(ox*ox+oy*oy)
            r = (i + j - math.fabs(i - j)) / 2  # min(i,j)
            s = (i + j + math.fabs(i - j)) / 2  # max(i,j)
            drn = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]
            forcex+=ox*((drn-dr2)/dr2)
            forcey+=oy*((drn-dr2)/dr2)
        next_step[i][0]=[current[i][0]-forcex*lr/size]
        next_step[i][1]=[current[i][1]-forcey*lr/size]

def execute_fmin(distance_matrix, projection, max_it, learning_rate0=0.5, lrmin= 0.05, decay=0.95, scale):
    size = len(projection)
    three_d = [from_2d(it) for it in projection]

    for i in prange(max_it):
        learning_rate = max(learning_rate0 * math.pow((1 - k / max_it), decay), lrmin)
        iterate_fmin(distance_matrix, np.copy(three_d), three_d, lr, scale)

    return [to_2d(it) for it in three_d]


