# https://en.wikipedia.org/wiki/Monotone_cubic_interpolation

#used for material thickness

import numpy as np
from typing import Tuple
import math

from MCHI_cu_wrapper import cu_interpolate_Ts

def _calc_M(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    if X.size != Y.size:
        print("sizes of X and Y need to match")
        
    #slopes of the secant lines between successive points
    S = np.zeros(X.size - 1)
    for k, S_i in enumerate(S):
        dy      = Y[k+1] - Y[k]
        dx      = X[k+1] - X[k]
        if dx == 0.0:
            S[k] = 3.0
        else:
            S[k] = dy / dx
    
    #print("S: ", S)
    
    #degree-1 coefficients at each point
    #Initialize tangents 
    #the ends come from S (one sided difference)
    #the interior points are the average
    #unless the signs of S change
    M               = np.zeros(X.size)
    M[0]            = S[0]
    M[M.size - 1]   = S[-1]
    for k in range(1, M.size - 2):
        if S[k] * S[k-1] <= 0:
            #if S_k = 0 then set M_k and M_k+1 = 0
            #print("Secant line negative or zero, S[k]= ", S[k], " S[k-1]= ", S[k-1])
            M[k] = 0
        else:
            M[k] = (S[k-1] + S[k]) / 2
    
    #calc the 2nd and 3rd order coefficients
    #with a bunch of conditions to ensure monotonicity
    for k in range(0, S.size):
        alpha_k    = 0
        beta_k     = 0
        if S[k] != 0:
            alpha_k    = M[k] / S[k]
            beta_k     = M[k+1] / S[k]
        #make piecewise monotone curve if needed
        if alpha_k < 0:
            M[k]    = 0
            #print("alpha_k < 0")
        elif alpha_k > 3:
            M[k]    = 3.0 * S[k]
            #print("alpha_k > 3")
            
        #make piecewise monotone curve if needed
        if beta_k < 0:
            M[k+1]  = 0
            #print("beta_k < 0")
        #restrict to a circle of radius 3        
        elif beta_k > 3:
            M[k+1]  = 3.0 * S[k]
            #print("beta_k > 3")
    
    return M

# https://en.wikipedia.org/wiki/Cubic_Hermite_spline
def _h_00(t):
    return 2*t*t*t - 3*t*t + 1
def _h_10(t):
    return t*t*t - 2*t*t + t
def _h_01(t):
    return -2*t*t*t + 3*t*t
def _h_11(t):
    return t*t*t - t*t

# see calling function for description
def _interpolate_Ts(Ts: np.ndarray, T_interpolated: np.ndarray, Ts_k_indices: np.ndarray):
    T_scale = Ts[-1] - Ts[0]
    N = T_interpolated.size
    for n in range(N):
        T_intrpltd_val = ((n / (N - 1)) * T_scale) + Ts[0] 
        
        k = 0
        for i in range(Ts.size):
            if T_intrpltd_val > Ts[i]:
                k = i
                
        Ts_k_indices[n]     = k
        T_interpolated[n]   = T_intrpltd_val

#see calling function for description
def _interpolate_Ys(Ts: np.ndarray, Ys: np.ndarray, Ms: np.ndarray, 
                     Ts_k_indices: np.ndarray, T_interpolated: np.ndarray, Y_interpolated: np.ndarray):
    N = T_interpolated.size
    for n in range(N):
        k       = Ts_k_indices[n]
        delta_t = Ts[k + 1] - Ts[k]
        t       = (T_interpolated[n] - Ts[k]) / delta_t
        Y_interpolated[n] = Ys[k]*_h_00(t) + delta_t*Ms[k]*_h_10(t) + Ys[k + 1]*_h_01(t) + delta_t*Ms[k + 1]*_h_11(t)

# Ts is the time or length axis to be interpolated.  For monotonic cubic interpolation it is expexted to be ordered. could be thought of as X
# Ys are the measures to be interpolated.
# N is the length of the desired interpolation
# return a tuple of interpolated Ts and interpolated Ys
def interpolate_1d_grid(Ts: np.ndarray, Ys: np.ndarray, N: int, use_cuda=False) -> Tuple[np.ndarray, np.ndarray]: 
    # first interpolate Ts
    # again, the incoming Ts are assumed to be ordered :-/
    # also find the left index into Ts
    # T_start = Ts[0]
    # T_end   = Ts[-1]
    # there are some manipulations (+Ts[0] and / (N -1)) to make sure we go [Ts[0] -> Ts[-1]]
    # for every value in T_interpolated, we find the value in Ts that is the largest withough going over
    # this is used in the interpolation
    T_interpolated = np.zeros(N)    
    Ts_k_indices = np.zeros(N, dtype=np.int32)
    if use_cuda == True:
        cu_interpolate_Ts(Ts, T_interpolated, Ts_k_indices)
    else:
        _interpolate_Ts(Ts, T_interpolated, Ts_k_indices)
    
    # we will first check if Y is multidimensional
    # if it is we will interpolate along the sequence for each dimension of the last 
    # Ms is what makes this interpolation monotonic
    # interpolate Ys
    # use the k (left) and k+1 (right) indices per n, into Ts, to calculate delta_t
    # calculate t within a section, which should be in the range [0->1]
    # calculate Y_interpolated per wikipedia
    if len(Ys.shape) == 1:
        Ms              = _calc_M(Ts, Ys)
        Y_interpolated  = np.zeros(N)
        _interpolate_Ys(Ts, Ys, Ms, Ts_k_indices, T_interpolated, Y_interpolated)
        return T_interpolated, Y_interpolated
    elif len(Ys.shape) == 2:
        Y_interpolated = np.zeros([N, Ys.shape[-1]])
        for feat in range(Ys.shape[-1]):
            Ms_feat = _calc_M(Ts, Ys[:,feat])
            _interpolate_Ys(Ts, Ys[:,feat], Ms_feat, Ts_k_indices, T_interpolated, Y_interpolated[:,feat])
        return T_interpolated, Y_interpolated
    else:
        what = input("aw naw")
        return [0], [0]

if __name__ == "__main__":
    
    import random

    for i in range(10):
        T_unsorted = np.random.rand(16)
        Ts = np.sort(T_unsorted)        #should be sorted in time
        Ys = np.random.rand(Ts.size,3)
        N = random.randint(256, 1024)
        
        T_interpolated, Y_interpolated = interpolate_1d_grid(Ts, Ys, N, use_cuda=False)
        print("Resolution= ", N, " total.")
        #print("X= ", X)
        #print("Y= ", Y)
        #print("interpolated X= ", intX)
        #print("interpolated Y=", intY)
        
        import matplotlib.pyplot as plt
        #plt.style.use('_mpl-gallery')
        
        fig, ax = plt.subplots(len(Ys[-1]), 1, constrained_layout=True)
        for i in range(len(Ys[-1])):
            ax[i].plot(Ts, Ys[:,i], linewidth = 2.0, color = "blue")
            ax[i].plot(T_interpolated, Y_interpolated[:, i], color = "red")
        
        plt.show()