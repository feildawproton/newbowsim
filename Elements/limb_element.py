#small angle approximation
import numpy as np
import math
from typing import Tuple

ALPHA = 1/50 #constant for mass assigned to thetas

def M_limb(rho_list: np.ndarray, w: float, h_list: np.ndarray, L: float) -> np.ndarray:
    linear_density = 0
    for j, rho_j in enumerate(rho_list):
        w_j             = w
        h_j             = h_list[j]
        A_j             = w_j * h_j
        linear_density += rho_j * A_j

    tot_mass = linear_density * L

    M_dist   = np.array([[1/2, 0,   0,         0,   0,   0        ],
                         [0,   1/2, 0,         0,   0,   0        ],
                         [0,   0,   ALPHA*L*L, 0,   0,   0        ],
                         [0,   0,   0,         1/2, 0,   0        ],
                         [0,   0,   0,         0,   1/2, 0        ],
                         [0,   0,   0,         0,   0,   ALPHA*L*L]])

    M        = tot_mass * M_dist
    return M

def _elastic_constants(w: float, h_list: np.ndarray, E_list: np.ndarray) -> Tuple[float, float, float]:

    I_list          = np.zeros(E_list.size)
    y_list          = np.zeros(E_list.size)

    current_h  = 0
    for j, h_j in enumerate(h_list):
        h_j        = h_list[j]
        A_j        = w * h_j

        y_j        = (h_j / 2) + current_h
        y_list[j]  = y_j
        
        I_j        = A_j * ( (h_j * h_j / 12) + (y_j * y_j) )
        I_list[j]  = I_j 

        current_h += h_j

    C_ee = 0
    C_kk = 0
    C_ek = 0
    
    for j, E_j in enumerate(E_list):
        h_j   = h_list[j]
        w_j   = w
        A_j   = w_j * h_j
        
        C_ee += E_j * A_j

        I_j   = I_list[j]
        C_kk += E_j * I_j

        y_j   = y_list[j]
        C_ek -= E_j * A_j * y_j

    return C_ee, C_kk, C_ek

def K_limb(w: float, h_list: np.ndarray, E_list: np.ndarray, L: float) -> np.ndarray:
    C_ee, C_kk, C_ek = _elastic_constants(w, h_list, E_list)
    
    X  =      C_ee / (L)
    Y1 = 12 * C_kk / (L * L * L) 
    Y2 =  6 * C_kk / (L * L)
    Y3 =  4 * C_kk / (L)
    Y4 =  2 * C_kk / (L)
    Z  =      C_ek / (L)

    K = np.array([[X,  0,   Z,  -X,  0,  -Z ],
                 [0,  Y1,  Y2,  0, -Y1,  Y2],
                 [Z,  Y2,  Y3, -Z, -Y2,  Y4],
                 [-X, 0,  -Z,   X,  0,   Z ],
                 [0, -Y1, -Y2,  0,  Y1, -Y2],
                 [-Z, Y2,  Y4,  Z, -Y2,  Y3]])
    return K

def p_limb(M: np.ndarray, K: np.ndarray, ddu: np.ndarray, u: np.ndarray) -> np.ndarray:
    F = np.matmul(M, ddu)
    q = np.matmul(K, u)
    p = F + q
    return p

#where the vector xy_0 -> xy_1 defines the rotation
def T_g2l_limb(x_0: float, y_0: float, x_1: float, y_1: float) -> np.ndarray:
    dx = x_1 - x_0
    dy = y_1 - y_0
    L = math.sqrt(dx * dx + dy * dy)

    c = dx / L
    s = dy / L

    T = np.array([[ c, s, 0,  0, 0, 0],
                  [-s, c, 0,  0, 0, 0],
                  [ 0, 0, 1,  0, 0, 0],
                  [ 0, 0, 0,  c, s, 0],
                  [ 0, 0, 0, -s, c, 0],
                  [ 0, 0, 0,  0, 0, 1]])

    return T

#where the vector xy_0 -> xy_1 defines the rotation
def T_l2g_limb(x_0: float, y_0: float, x_1: float, y_1: float) -> np.ndarray:
    dx = x_1 - x_0
    dy = y_1 - y_0
    L = math.sqrt(dx * dx + dy * dy)

    c = dx / L
    s = dy / L

    T = np.array([[c, -s, 0, 0,  0, 0],
                  [s,  c, 0, 0,  0, 0],
                  [0,  0, 1, 0,  0, 0],
                  [0,  0, 0, c, -s, 0],
                  [0,  0, 0, s,  c, 0],
                  [0,  0, 0, 0,  0, 1]])

    return T
    
if __name__ == "__main__":
    print("using small angle approximation of beam aligned along x direction")
    print("rotation angles assumed to be the derivative w' of the bending line")
    print("u_0 = u_0 = x displacement of first node")
    print("u_1 = v_0 = y displacement of the first node")
    print("u_2 = %theta_0 = angle off of x direction of the first node")
    print("u_3 = u_1 = x displacement of the second node")
    print("U_4 = V_1 = y displacement of the second node")
    print("u_5 = %theta_1 = angle off the x direction of the second node")
    #import beam
    import random
    for i in range(10):
        n_layers = random.randint(1, 10)
        rho_list = np.random.rand(n_layers)
        w_list   = np.random.rand(n_layers)
        h_list   = np.random.rand(n_layers)
        L        = random.random()
        E_list   = np.random.rand(n_layers)
        
        print("Mass matrix")
        M = M_limb(rho_list, w_list, h_list, L)
        print(M)
        #print("compare to previous implementation")
        #M = beam.M_beam(rho_list, w_list, h_list, L)
        #print(M)

        print("elastic constants")
        C_ee, C_kk, C_ek = _elastic_constants(w_list, h_list, E_list)
        print(C_ee, C_kk, C_ek)
        #print("compare to other implementation")
        #C_ee, C_kk, C_ek = beam.elastic_constants(E_list, w_list, h_list)
        #print(C_ee, C_kk, C_ek)

        print("Stiffness matrix")
        K = K_limb(w_list, h_list, E_list, L)
        print(K)
        #print("check other implementation")
        #C = beam.C_beam(E_list, w_list, h_list, L)
        #print(C) 
        #print("difference:")
        #print(K-C)

        u_0 = random.random()
        v_0 = random.random()
        w_0 = random.random()
        u_1 = random.random()
        v_1 = random.random()
        w_1 = random.random()

        ddu_0 = random.random()
        ddv_0 = random.random()
        ddw_0 = random.random()
        ddu_1 = random.random()
        ddv_1 = random.random()
        ddw_1 = random.random()

        u = np.array([u_0, v_0, w_0, u_1, v_1, w_1])
        ddu = np.array([ddu_0, ddv_0, ddw_0, ddu_1, ddv_1, ddw_1])

        p = p_limb(M, K, ddu, u)
        print("the p vector: ", p)

        x_0 = random.random()
        y_0 = random.random()
        x_1 = random.random()
        y_1 = random.random()

        T = T_g2l_limb(x_0, y_0, x_1, y_1)
        TT = T.transpose()
        print("transform")
        print(T)
        print("transpose")
        print(TT)

        T = T_g2l_limb(x_0, y_0, x_1, y_1)
        TT = T.transpose()       
        T_l2g = T_l2g_limb(x_0, y_0, x_1, y_1)
        print("transpose global to local to local to global")
        print(TT)
        print("local to global matrix")
        print(T_l2g)
        print("these should be the same:")
        print(TT - T_l2g) 

        what = input("observe and pontificate")