import numpy as np
from typing import Tuple

from Elements.limb_element import T_g2l_limb

#calculate an array of T that transform from global to local coordinates
def calc_T_g2l_limb(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    assert X.size == Y.size
    #T_g2l describes how to rotate the reference frame back into global
    #applying this transform to a vector in global will result in the vector as it appears to the local transform
    T_limb = np.zeros([X.size - 1, 6, 6])
    for i in range(T_limb.shape[0]):
        T           = T_g2l_limb(X[i], Y[i], X[i+1], Y[i+1])
        T_limb[i]   = T
    return T_limb

#an array of K' and an array of Ts
#Kp means K' with are local
def Ms_l2g(Kp: np.ndarray, Ts_g2l: np.ndarray) -> np.ndarray:
    assert Kp.shape == Ts_g2l.shape
    #if you a transform, T, that describes how to rotate a reference frame back to global
    #the it is a global to local transflorm
    #it could be use to transform global u into local for work with local K'
    #but in this case we can transform local K into global with T^{T} K' T = K 
    #were just doing that for an array of transforms and K's
    K = np.zeros(Kp.shape)
    for i in range(K.shape[0]):
        T_i     = Ts_g2l[i]
        TT_i    = Ts_g2l[i].transpose()
        Kp_i    = Kp[i]
        right   = np.matmul(Kp_i, T_i)
        K[i]    = np.matmul(TT_i, right)
    return K

#gK is the global matrix
#gKs_limb is a list (array) of limb stiffness matrices in global coordinates
#gK should be big enough to take the limb and then some perhaps (like the free string)
#this is separate function because inserts of the limb into gK is very straight forward
def insert_gKlimb(gK: np.ndarray, gKs_limb: np.ndarray):
    n_elems = gKs_limb.shape[0]
    n_nodes = n_elems + 1
    assert n_nodes * 3 >= gK.shape[0] == gK.shape[1]
    for i in range(n_elems):
        gindx = 3 * i
        gK[gindx:gindx+6, gindx:gindx+6] = np.add(gK[gindx:gindx+6, gindx:gindx+6], gKs_limb[i])
        
def update_pos(X: np.ndarray, Y: np.ndarray, u_i: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #sanity check on the array sizes
    #I know I do this a lot but these can get away from you easily
    #note the mixed use of .size and .shape[0]
    #this is to ensure that they have the sam meaning in the context of these arrays
    num_nodes = X.size
    assert num_nodes * 3 == Y.shape[0] * 3 == u_i.size 
    
    #a simple mapping and addition
    #the array u_i is organize [u_0, v_0, theta_0, ... , u_N-1, v_N-1, theta_N-1]
    #where u is change in x, v is change in y, and theta is change in angle
    X_new = np.zeros(num_nodes)
    Y_new = np.zeros(num_nodes)
    for i in range(num_nodes):
        X_new[i] = X[i] + u_i[i*3]
        Y_new[i] = Y[i] + u_i[i*3 + 1]
    
    return X_new, Y_new