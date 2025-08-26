import numpy as np
import math
from numba import njit, prange
import argparse
import os
import math


@njit(parallel=True)
def iterate_fmin(distance_matrix, current, next_step, lr):
    size = len(current)
    total = int(size*(size+1)/2)
    for i in prange(size):
        forcex=0
        forcey=0
        for j in prange(size):
            if(i==j):
                continue
            ox = current[j][0] - current[i][0]
            oy = current[j][1] - current[i][1]
            dr2 = max(math.sqrt(ox*ox+oy*oy), 0.0001)
            r = (i + j - math.fabs(i - j)) / 2  # min(i,j)
            s = (i + j + math.fabs(i - j)) / 2  # max(i,j)
            drn = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]
            forcex+=ox*((drn-dr2)/dr2)
            forcey+=oy*((drn-dr2)/dr2)
        next_step[i][0]=current[i][0]-forcex*lr/size
        next_step[i][1]=current[i][1]-forcey*lr/size


def execute_fmin(distance_matrix, projection, max_it, learning_rate0=0.5, lrmin= 0.05, decay=0.95):
    size = len(projection)
    for i in prange(max_it):
        learning_rate = max(learning_rate0 * math.pow((1 - i / max_it), decay), lrmin)
        iterate_fmin(distance_matrix, np.copy(projection), projection, learning_rate)
        print("Iteration: "+str(i)+"/"+str(max_it))
        
    # setting the min to (0,0)
    min_x = min(projection[:, 0])
    min_y = min(projection[:, 1])
    for i in range(size):
        projection[i][0] -= min_x
        projection[i][1] -= min_y



def main():
    """ Read the result of ELViM projection, minimize the forces and write to a different file.
           
    Args:
        -dm (filename): File containing a dissimilarity matrix (default: dmat.npy)
        -it (int): Maximum number of iteration (default: sqrt(n_frames))
        -lr0 (float): Initial value for the learning_rate (default: 1.1)
        -lrmin (float): Minimum value for the learning_rate (default: 1)
        -d (float): Exponential decay of the learning_rate (default: 0.95)
        -i (filename): ELViM Projection coordinates file (default: projection.out)
        -o (filename): Name to save the minimized forces coordinates file (default: fminimized.out)
    Returns:
        output (file): Minimized forces coordinates file (default: fminimized.out)
    """
    
    message='''This program can read a distance matrix in the symmetrical square or condensed format and the result of an
           ELViM projection to minimize residual forces in the projection and output as a different file.'''
        
    parser = argparse.ArgumentParser(description=message)
    parser.add_argument("-dm", dest="dmat", 
                  action="store", type=str, default="dmat.npy",
                  help="Name of the dissimilarity matrix file")
    parser.add_argument("-it", dest="max_it", 
                  action="store", type=int, default=None,
                  help="Number of iterations (default sqrt(n_frames")
    parser.add_argument("-lr0", dest="learning_rate0", action="store", type=float, default=1.1,
                  help="Initial learning rate")
    parser.add_argument("-lrmin", dest="learning_rate_min", action="store", type=float, default=1,
                  help="Minimum value for learning rate")
    parser.add_argument("-d", dest="decay", action="store", type=float, default=0.95,
                  help="Learning rate exponential decay")
    parser.add_argument("-i", dest="input_proj",
                  action="store", type=str, default="projection.out",
                  help="ELViM projection coordinates")
    parser.add_argument("-o", dest="output_proj",
                  action="store", type=str, default="fminimized.out",
                  help="FMin coordinates")


   
    opt = parser.parse_args()
    dmat_file=opt.dmat
    max_it=opt.max_it
    lr0=opt.learning_rate0
    lrmin=opt.learning_rate_min
    decay=opt.decay
    inp=opt.input_proj
    outp=opt.output_proj

    if not os.path.isfile(dmat_file):
            raise RuntimeError('''Failed to load the dissimilarity matrix(file doesn't exist), aborting.''')
    if not os.path.isfile(inp):
            raise RuntimeError('''Failed to load the ELViM projection(file doesn't exist), aborting.''')
    
    dmat = np.load(dmat_file)# read_dmat(dmat_file)
    size = int(np.sqrt(2 * len(dmat)))
    print('Dissimilaty matrix loaded. There are {} conformations'.format(size))
    
  
    ### Make the projection
    ### Initializing the projection with points randomly distributed
    projection = None
    with open(inp, 'r') as f:
        projection = np.loadtxt(f)

    if(len(projection) != size):
            raise RuntimeError("ELViM projection has different size than the dissimilarity matrix ({} != {}).".format(len(projection), size))
    
    if max_it==None:
        max_it = int(np.sqrt(size))
    
    ### Running force minimization
    execute_fmin(dmat, projection, max_it, lr0, lrmin, decay)
   
    ### write the projection coordinates file
    with open(outp, 'w') as f:
       f.write("# FMin Coordinates file. \n") 
       f.write("# Dissimilarity matrix read from the file : {}\n".format(dmat_file))
       f.write("# {} iterations with lr0 = {:5.3f}, lrmin = {:5.3f}, and decay = {:5.3f} \n".format(max_it, lr0, lrmin, decay))
       np.savetxt(f, projection, delimiter=" ", fmt="%.4f")
    
if __name__=="__main__":
	main()
	exit(0)