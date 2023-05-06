# my guess at interpolating 
import numpy as np
from typing import Tuple
import math

def _calc_Ds(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if X.size != Y.size:
        print("sizes need to match")
        return [0], [0]
    else:
        DXs = np.zeros(X.size)
        DYs = np.zeros(Y.size)
        for k in range(1, DXs.size - 1):
            DXs[k] = X[k+1] - X[k-1]
            DYs[k] = Y[k+1] - Y[k-1]
            '''
            mag = math.sqrt(dx*dx + dy*dy)
            MXs[k] = dx / mag
            MYs[k] = dy / mag
            '''
        return DXs, DYs
    

# https://en.wikipedia.org/wiki/Cubic_Hermite_spline
def _h_00(t):
    return 2*t*t*t - 3*t*t + 1
def _h_10(t):
    return t*t*t - 2*t*t + t
def _h_01(t):
    return -2*t*t*t + 3*t*t
def _h_11(t):
    return t*t*t - t*t

def _interpolate_section(x_k, x_kp1, y_k, y_kp1, Dx_k, Dx_kp1, Dy_k, Dy_kp1, N) -> Tuple[np.ndarray, np.ndarray]:
    '''
    dx = x_kp1 - x_k
    dy = y_kp1 - y_k
    mag = math.sqrt(dx*dx + dy*dy)
    '''
    
    X = np.zeros(N)
    Y = np.zeros(N)
    X[0] = x_k
    Y[0] = y_k
    for n in range(1, N):
        t = n / N
        Y[n] = y_k*_h_00(t) + Dy_k*_h_10(t) + y_kp1*_h_01(t) + Dy_kp1*_h_11(t)
        X[n] = x_k*_h_00(t) + Dx_k*_h_10(t) + x_kp1*_h_01(t) + Dx_kp1*_h_11(t)
    
    return X, Y

def _interpolate(X: np.ndarray, Y: np.ndarray, DXs: np.ndarray, DYs: np.ndarray, Ns: np.array) -> Tuple[np.ndarray, np.ndarray]:
    if X.size != Y.size != M.size:
        print("lengths of X, Y, and M must be the same")

    intX = []
    intY = []
    for k in range(X.size - 1):
        X_section, Y_section = _interpolate_section(X[k], X[k+1], Y[k], Y[k+1], 
                                                   DXs[k], DXs[k+1], DYs[k], DYs[k+1],
                                                   Ns[k])
        intX = np.append(intX, X_section)
        intY = np.append(intY, Y_section)
        
    intX = np.append(intX, X[-1])
    intY = np.append(intY, Y[-1])
    return intX, intY

def interpolate_2d(X: np.ndarray, Y: np.ndarray, N_tot: int) -> Tuple[np.ndarray, np.ndarray]:
    print("Requesting Ns: ", N_tot)
    #proportially assign the sections
    #print("Number of Ls= ",X.size)
    tot_len = 0
    lens = np.zeros(X.size - 1)
    for i in range(lens.size):
        length   = math.sqrt((Y[i + 1] - Y[i])*(Y[i + 1] - Y[i]) + (X[i + 1] - X[i])*(X[i + 1] - X[i]))
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

    DXs, DYs = _calc_Ds(X, Y)
    #print("num DX ", DXs.size)
    #print("Num DY ", DYs.size)
    intX, intY = _interpolate(X, Y, DXs, DYs, Ns)
    return intX, intY

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    import random
    for n in range(2, 10):
        
        X = np.random.rand(n)
        Y = np.random.rand(n)
        N = random.randint(n, n*16)       
        
        intX, intY = interpolate_2d(X, Y, N)
        print("Resolution= ", N, " total.")
        '''
        print("X= ", X)
        print("Y= ", Y)
        print("interpolated X= ", intX)
        print("interpolated Y=", intY)
        '''
        
        fig, ax = plt.subplots()
        
        ax.plot(X, Y, linewidth = 2.0, color = "blue")
        ax.plot(intX, intY, color = "red")
        
        plt.show()

