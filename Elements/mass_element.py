#for example the arrow
import numpy as np
import math

def M_point(m: float) -> np.ndarray:
    M_dist = np.array([[1, 0],
                       [0, 1]])
    M = m * M_dist
    return M

def K_point() -> np.ndarray:
    K = np.array([[0, 0],
                  [0, 0]])
    return K
    
def p_point(M: np.ndarray, ddu: np.ndarray) -> np.ndarray:
    F = np.matmul(M, ddu)
    return F

def T_g2l_point(x_0: float, y_0: float, x_1: float, y_1: float) -> np.ndarray:
    dx = x_1 - x_0
    dy = y_1 - y_0
    L = math.sqrt(dx * dx + dy * dy)

    c = dx / L
    s = dy / L

    T = np.array([[ c, s],
                  [-s, c]])

    return T

def T_l2g_point(x_0: float, y_0: float, x_1: float, y_1: float) -> np.ndarray:
    dx = x_1 - x_0
    dy = y_1 - y_0
    L = math.sqrt(dx * dx + dy * dy)

    c = dx / L
    s = dy / L

    T = np.array([[c, -s],
                  [s,  c]])

    return T

if __name__ == "__main__":
    import random
    #import bar
    for i in range(10):
        m = random.random()
        M = M_point(m)
        print("mass: ", m)
        print(M)

        ddu_0 = random.random()
        ddv_0 = random.random()
        ddu   = np.array([ddu_0, ddv_0])

        p = p_point(M, ddu)
        print("p = F + 0 matrix")
        print(p)

        x_0 = random.random()
        y_0 = random.random()
        x_1 = random.random()
        y_1 = random.random()

        T = T_g2l_point(x_0, y_0, x_1, y_1)
        dx = x_1 - x_0
        dy = y_1 - y_0
        mag = math.sqrt(dx*dx + dy*dy)
        u = dx / mag
        v = dy / mag
        gvec = np.array([u,v])
        lvec = np.matmul(T, gvec)
        print("global vec: ", gvec)
        print("local vec: ", lvec)

        T = T_g2l_point(x_0, y_0, x_1, y_1)
        TT = T.transpose()       
        T_l2g = T_l2g_point(x_0, y_0, x_1, y_1)
        print("transpose global to local to local to global")
        print(TT)
        print("local to global matrix")
        print(T_l2g)
        print("these should be the same:")
        print(TT - T_l2g) 

        what = input("observe and confirm")