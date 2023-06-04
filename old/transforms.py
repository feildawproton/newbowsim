import numpy as np
import math

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

def T_g2l_vec(x_0: float, y_0: float, x_1: float, y_1: float) -> np.ndarray:
    dx = x_1 - x_0
    dy = y_1 - y_0
    L = math.sqrt(dx * dx + dy * dy)

    c = dx / L
    s = dy / L

    T = np.array([[ c, s],
                  [-s, c]])

    return T

def T_l2g_vec(x_0: float, y_0: float, x_1: float, y_1: float) -> np.ndarray:
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

    for i in range(10):
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
        
        T = T_g2l_vec(x_0, y_0, x_1, y_1)
        dx = x_1 - x_0
        dy = y_1 - y_0
        mag = math.sqrt(dx*dx + dy*dy)
        u = dx / mag
        v = dy / mag
        gvec = np.array([u,v])
        lvec = np.matmul(T, gvec)
        print("global vec: ", gvec)
        print("local vec: ", lvec)

        T = T_g2l_limb(x_0, y_0, x_1, y_1)
        TT = T.transpose()       
        T_l2g = T_l2g_limb(x_0, y_0, x_1, y_1)
        print("transpose global to local to local to global")
        print(TT)
        print("local to global matrix")
        print(T_l2g)
        print("these should be the same:")
        print(TT - T_l2g) 

        T = T_g2l_vec(x_0, y_0, x_1, y_1)
        TT = T.transpose()       
        T_l2g = T_l2g_vec(x_0, y_0, x_1, y_1)
        print("transpose global to local to local to global")
        print(TT)
        print("local to global matrix")
        print(T_l2g)
        print("these should be the same:")
        print(TT - T_l2g) 

        what = input("observe and despair")

