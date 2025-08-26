import numpy as np
import math
from numba import njit, prange
import argparse
import os
import math


@njit(fastmath=False)
def from_2d(point):
    s = math.sin(point[1])
    return [math.cos(point[0])*s, math.sin(point[0])*s, math.cos(point[1])]

#@njit(fastmath=False) # This actually stops working if it's @njit
def to_2d(point):
    magnitude = math.sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2])
    clamped = max(-1, min(1, point[2]/magnitude))
    result = [math.atan2(point[1], point[0]), math.acos(clamped)]
    #print(str(point))
    #print(str(result))
    return result

@njit(fastmath=False)
def slerp(A, B, t, omega):
    d = math.sin(omega)
    s0 = math.sin((1-t)*omega)
    s1 = math.sin(t*omega)
    x = (A[0]*s0 + B[0]*s1)
    y = (A[1]*s0 + B[1]*s1)
    z = (A[2]*s0 + B[2]*s1)
    return [x/d,y/d,z/d]

#def slerp(A, B, t):
#    dotA=A[0]*A[0]+A[1]*A[1]+A[2]*A[2]
#    dotB=B[0]*B[0]+B[1]*B[1]+B[2]*B[2]
#    dotAB=A[0]*B[0]+A[1]*B[1]+A[2]*B[2]
#    math.acos(dotAB/(math.sqrt(dotA)*math.sqrt(dotB)))
#    return slerp_unit_vector(A,B,t,omega)

@njit(parallel=True, fastmath=False)
def move_sph(ins1, distance_matrix, projection, learning_rate, radius):
    size = len(projection)
    total = len(distance_matrix)
    error = 0

    pointA = projection[ins1]

    for ins2 in prange(size):
        if ins1 == ins2:
            continue
        pointB = projection[ins2]

        dot = pointA[0]*pointB[0] + pointA[1]*pointB[1] + pointA[2]*pointB[2]
        #print(str(dot))
        dr2 = max(math.acos(dot), 0.01)

        # getting te index in the distance matrix and getting the value
        r = (ins1 + ins2 - math.fabs(ins1 - ins2)) / 2  # min(ins1,ins2)
        s = (ins1 + ins2 + math.fabs(ins1 - ins2)) / 2  # max(ins1,ins2)
        drn = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))] *radius

        # calculate the movement
        delta = (drn - dr2) #* math.fabs(drn - dr2)

        res = slerp(pointA,pointB,math.exp(learning_rate*delta/dr2),dr2)
        mag = math.sqrt(res[0]*res[0]+res[1]*res[1]+res[2]*res[2])
        # moving
        projection[ins2][0]=res[0]/mag
        projection[ins2][1]=res[1]/mag
        projection[ins2][2]=res[2]/mag

#@njit(parallel=True, fastmath=False)
def iteration_sph(index, distance_matrix, projection, learning_rate, radius):
    size = len(projection)
    for i in prange(size):
        ins1 = index[i]
        move_sph(ins1, distance_matrix, projection, learning_rate, radius)


def execute_sph(distance_matrix, projection, max_it, radius, learning_rate0=0.5, lrmin= 0.05, decay=0.95):
    size = len(projection)
    # create random index
    index = np.random.permutation(size)
    
    # convert from 2D coordinates to 3D
    three_d = np.array([from_2d(it) for it in projection])
    print("Shape: {}x{}".format(len(three_d), len(three_d[0])))

    for k in range(max_it):
            
        print("ELViM Sph Iteration: "+str(k)+"/"+str(max_it))
        learning_rate = max(learning_rate0 * math.pow((1 - k / max_it), decay), lrmin)
        iteration_sph(index, distance_matrix, three_d, learning_rate, radius)

        # For testing purposes
        #with open("test"+str(k), 'w') as f:
        #   f.write("# ELViM Spherical Coordinates file. \n")
        #   f.write("# Learning rate = {} \n".format(learning_rate))
        #   f.write("# This is only for debugging\n")
        #   np.savetxt(f, [to_2d(it) for it in three_d], delimiter=" ", fmt="%.4f")
        

    return [to_2d(it) for it in three_d]


@njit(parallel=True)
def compute_scaling_factor(distance_matrix, size):
    #total = 0
    #for k in prange (size):
    #    partial = 0
    #    for l in prange (k, size):
    #        partial += distance_matrix[int(total - ((size - k) * (size - k + 1) / 2) + (l - k))]
    #    total+=partial/(k-size)
    return (math.pi/2)/(np.sum(distance_matrix)/len(distance_matrix))


def main():
    """ Read a dissimilarity matrix to generate an ELViM projection on the surface of a sphere.
           
    Args:
        -dm (filename): File containing a dissimilarity matrix (default: dmat.npy)
        -it (int): Maximum number of iteration (default: sqrt(n_frames))
        -lr0 (float): Initial value for the learning_rate (default: 1.1)
        -lrmin (float): Minimum value for the learning_rate (default: 1)
        -d (float): Exponential decay of the learning_rate (default: 0.95)
        -radius (float): Radius of sphere
            Leave blank for the value to be calculated based on dissimilarity matrix

        -elvim (filename): ELViM Spherical Projection file (default: projection.sph.out)
        -fmin (filename): Minimized Forces Spherical Projection file (default: fminimized.sph.out)
    Returns:
        output (file): Minimized Forces Spherical Projection file (default: fminimized.sph.out)
    """
    
    message='''This program can read a distance matrix in the condensed format,
Generate a ELViM projection on a sphere(or load it, if it exists)
then optionally generates the force minimization'''
        
    parser = argparse.ArgumentParser(description=message)
    parser.add_argument("-dm", dest="dmat", 
                  action="store", type=str, default="dmat.npy",
                  help="Name of the dissimilarity matrix file")
    parser.add_argument("-it", dest="max_it", 
                  action="store", type=int, default=None,
                  help="Number of iterations (default sqrt(n_frames)//10)")
    parser.add_argument("-lr0", dest="learning_rate0", action="store", type=float, default=0.2,
                  help="Initial learning rate for force scheme")
    parser.add_argument("-lrmin", dest="learning_rate_min", action="store", type=float, default=0,
                  help="Minimum value for learning rate during force scheme")
    parser.add_argument("-d", dest="decay", action="store", type=float, default=1.5,
                  help="Learning rate exponential decay during force scheme")
    parser.add_argument("-radius", dest="radius", action="store", type=float, default=None,
                  help="Radius of sphere, leave blank for the value to be calculated based on dissimilarity matrix")
    parser.add_argument("-o", dest="elvim_proj",
                  action="store", type=str, default="projection.sph.out",
                  help="Filename to save the ELViM projection coordinates to")


   
    opt = parser.parse_args()
    dmat_file=opt.dmat
    max_it=opt.max_it
    lr0=opt.learning_rate0
    lrmin=opt.learning_rate_min
    decay=opt.decay
    radius = opt.radius
    elvim_proj=opt.elvim_proj

    if not os.path.isfile(dmat_file):
        raise RuntimeError('''Failed to load the dissimilarity matrix(file doesn't exist), aborting.''')
    
    dmat = np.load(dmat_file)# read_dmat(dmat_file)
    size = int(np.sqrt(2 * len(dmat)))
    print('Dissimilaty matrix loaded. There are {} conformations'.format(size))

    if radius == None:
        radius = compute_scaling_factor(dmat, size)
        print("Computed scaling factor: "+str(radius))
    else:
        print("Using provided scaling factor: "+str(radius))

    if max_it==None:
        max_it = int(np.sqrt(size))
  
    ### Make the projection
    ### Initializing the projection with points randomly distributed
    projection = None
    
    projection = np.random.random((size, 2))
    result = execute_sph(dmat, projection, max_it//10, radius, lr0, lrmin, decay)
    with open(elvim_proj, 'w') as f:
       f.write("# ELViM Spherical Coordinates file. \n") 
       f.write("# Dissimilarity matrix read from the file : {}\n".format(dmat_file))
       f.write("# This file contains the angular coordinates with radius {}\n".format(radius))
       f.write("# {} iterations with lr0 = {:5.3f}, lrmin = {:5.3f}, and decay = {:5.3f} \n".format(max_it, lr0, lrmin, decay))
       np.savetxt(f, result, delimiter=" ", fmt="%.4f")
    
if __name__=="__main__":
	main()
	exit(0)





