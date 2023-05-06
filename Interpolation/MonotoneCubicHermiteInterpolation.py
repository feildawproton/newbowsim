# https://en.wikipedia.org/wiki/Monotone_cubic_interpolation

#used for material thickness

import numpy as np
from typing import Tuple
import math

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

def _interpolate_section(x_k, x_kp1, y_k, y_kp1, m_k, m_kp1, res) -> Tuple[np.ndarray, np.ndarray]:
    delta = x_kp1 - x_k
    
    X = np.zeros(res)
    Y = np.zeros(res)
    for n in range(res):
        t = n / res
        Y[n] = y_k*_h_00(t) + delta*m_k*_h_10(t) + y_kp1*_h_01(t) + delta*m_kp1*_h_11(t)
        X[n] = x_k + t*delta
    
    return X, Y

def _interpolate(L: np.ndarray, W: np.ndarray, M: np.ndarray, Ns: int) -> Tuple[np.ndarray, np.ndarray]:
    if L.size != W.size != M.size:
        print("lengths of X, Y, and M must be the same")

    intX = []
    intY = []
    for k in range(L.size - 1):
        X_section, Y_section = _interpolate_section(L[k], L[k+1], W[k], W[k+1], M[k], M[k+1], Ns[k])
        intX = np.append(intX, X_section)
        intY = np.append(intY, Y_section)
    intX = np.append(intX, L[-1])
    intY = np.append(intY, W[-1])
    return intX, intY
    
def interpolate_1d(L: np.ndarray, W: np.ndarray, N_tot: int) -> Tuple[np.ndarray, np.ndarray]:
    #proportially assign the sections
    print("Requesting N_tot= ", N_tot)
    tot_len = 0
    lens = np.zeros(W.size - 1)
    for i in range(lens.size):
        length   = math.sqrt((W[i + 1] - W[i])*(W[i + 1] - W[i]) + (L[i + 1] - L[i])*(L[i + 1] - L[i]))
        #print(length)
        lens[i]  = length
        tot_len += length
    #print("total len= ", tot_len)
    #print("lens= ", lens)

    Ns = np.zeros(lens.size, dtype=int)
    new_n_tot = 0
    for i in range(Ns.size):
        frac        = lens[i] / tot_len
        N           = int(frac * N_tot) + 1
        Ns[i]       = N
        new_n_tot   += N
        #print(lens[i], N, new_n_tot)
    
    print("Ns: ", Ns)
    print("New Total N= ", new_n_tot)

    M = _calc_M(L, W)
    intL, intW = _interpolate(L, W, M, Ns)
    return intL, intW

if __name__ == "__main__":
    
    import random
    for i in range(10):
        L = np.array([0, 1,  2, 3,  4,  5,  6,   7])
        W = np.random.rand(8)
        N = random.randint(8, 64)
        
        intX, intY = interpolate_1d(L, W, N)
        print("Resolution= ", N, " total.")
        #print("X= ", X)
        #print("Y= ", Y)
        #print("interpolated X= ", intX)
        #print("interpolated Y=", intY)
        
        import matplotlib.pyplot as plt
        #plt.style.use('_mpl-gallery')
        
        fig, ax = plt.subplots()
        
        ax.plot(L, W, linewidth = 2.0, color = "blue")
        ax.plot(intX, intY, color = "red")
        
        plt.show()