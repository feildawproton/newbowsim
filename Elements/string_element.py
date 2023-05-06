#a bar element has two nodes A and B
import numpy as np
import math

CHAR_BAR = np.array([[1, 0, -1, 0],
                     [0, 1, 0, -1],
                     [-1, 0, 1, 0],
                     [0, -1, 0, 1] ])

def M_string(rho: float, A: float , L: float) -> np.ndarray:
    tot_mass = rho * A * L / 2
    mass_dist = np.array([[1,0,0,0],
                          [0,1,0,0],
                          [0,0,1,0],
                          [0,0,0,1]])
    M = tot_mass * mass_dist
    return M

#tangent stiffness matrix
def K_string(E: float, A: float, L: float, u_0: float, v_0: float, u_1: float, v_1: float) -> np.ndarray:
    elastic_const = E * A / L
    #print("New elastic const: ", elastic_const)
    dx = u_1 - u_0
    dy = v_1 - v_0
    mag = math.sqrt(dx*dx + dy*dy)
    elastic_scale = 1 - L / mag
    #print("new elastic_scale: ", elastic_scale)

    K = elastic_const * elastic_scale * CHAR_BAR
    return K

#Tangent damping matrix
def D_string(eta: float, A: float, L: float) -> np.ndarray:
    viscous_const = eta * A / L
    D = viscous_const * CHAR_BAR
    return D

def p_string(M: np.ndarray, D: np.ndarray, K: np.ndarray, ddu: np.ndarray, du: np.ndarray, u: np.ndarray) -> np.ndarray:
    F = np.matmul(M, ddu)
    q = np.matmul(D, du) + np.matmul(K, u)
    p = F + q
    return p

#where the vector xy_0 -> xy_1 defines the rotation
def T_g2l_string(x_0: float, y_0: float, x_1: float, y_1: float) -> np.ndarray:
    dx = x_1 - x_0
    dy = y_1 - y_0
    L = math.sqrt(dx * dx + dy * dy)

    c = dx / L
    s = dy / L

    T = np.array([[ c, s,  0, 0],
                  [-s, c,  0, 0],
                  [ 0, 0,  c, s],
                  [ 0, 0, -s, c]])

    return T

#where the vector xy_0 -> xy_1 defines the rotation
def T_l2g_string(x_0: float, y_0: float, x_1: float, y_1: float) -> np.ndarray:
    dx = x_1 - x_0
    dy = y_1 - y_0
    L = math.sqrt(dx * dx + dy * dy)

    c = dx / L
    s = dy / L

    T = np.array([[c, -s, 0,  0],
                  [s,  c, 0,  0],
                  [0,  0, c, -s],
                  [0,  0, s,  c]])
                  
    return T

if __name__ == "__main__":
    import random
    #import bar
    for i in range(10):
        E = random.random()
        A = random.random()
        L = random.random()
        rho = random.random()
        eta = random.random()

        u_0 = random.random()
        v_0 = random.random()
        u_1 = random.random()
        v_1 = random.random()

        print("Mass matrix")
        M = M_string(rho, A, L)
        print(M)

        print("Tangent stiffness matrix: ")
        K = K_string(E, A, L, u_0, v_0, u_1, v_1)
        print(K)
        '''
        print("old K difference")
        K_old = bar.K_bar(E, A, L, u_0, v_0, u_1, v_1)
        print(K - K_old)
        '''

        print("tangent damping matrix: ")
        D = D_string(eta, A, L)
        print(D)
        '''
        print("old D difference")
        D_old = bar.D_bar(eta, A, L)
        print(D - D_old)
        '''

        du_0 = random.random()
        dv_0 = random.random()
        du_1 = random.random()
        dv_1 = random.random()

        ddu_0 = random.random()
        ddv_0 = random.random()
        ddu_1 = random.random()
        ddv_1 = random.random()

        u = np.array([u_0, v_0, u_1, v_1])
        du = np.array([du_0, dv_0, du_1, dv_1])
        ddu = np.array([ddu_0, ddv_0, ddu_1, ddv_1])

        p = p_string(M, D, K, ddu, du, u)
        print("P new: ", p)
        '''
        P_0, P_1, P_2, P_3 = bar.p_bar(rho, A, L, eta, E, 
                                        ddu_0, ddv_0, ddu_1, ddv_1,
                                        du_0, dv_0, du_1, dv_1,
                                        u_0, v_0, u_1, v_1)
        print("Old p: ", P_0, P_1, P_2, P_3)
        '''

        x_0 = random.random()
        y_0 = random.random()
        x_1 = random.random()
        y_1 = random.random()

        T = T_g2l_string(x_0, y_0, x_1, y_1)
        TT = T.transpose()       
        T_l2g = T_l2g_string(x_0, y_0, x_1, y_1)
        print("transpose global to local to local to global")
        print(TT)
        print("local to global matrix")
        print(T_l2g)
        print("these should be the same:")
        print(TT - T_l2g) 

        what = input("LOOK at the output, press a key")
    
    






