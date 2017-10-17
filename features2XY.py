# convert features and counts matrix to the format of X, Y
# input 
#   features, patch images' feature extracted from ResNet;
#   counts, patch images' count;
# output
#   X, the input of the fully connected regress network with the dimension of M x 1000, which 
#   M is the number of the training image patches in formula (1);
#   Y, the output of the fully connected regress network with the dimension of M x 1; 

import numpy as np 

def features2XY(features, counts):
    n = 0
    for c in counts:
        n = n + c.size

    X = np.zeros((n, 1000))
    Y = np.zeros((n, 1))
    k = 0
    for (patch_feature, patch_count) in zip(features, counts):
        X[k:k + patch_count.size, :] = patch_feature.reshape(patch_count.size, 1000)
        Y[k:k + patch_count.size] = patch_count.reshape(patch_count.size, 1)
        k = k + patch_count.size

    return X, Y
