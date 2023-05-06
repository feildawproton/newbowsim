import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Tuple

from Interpolation.spline2d import interpolate_2d
from BowModel.Settings import General
from BowModel.Profile import Spline
from BowModel.Width import Width
from Interpolation.MonotoneCubicHermiteInterpolation import interpolate_1d
from BowModel.Layers import HickoryLayer
from BowModel.Materials import Hickory

from Elements.limb_element import M_limb, K_limb, T_g2l_limb


def _curved_len(X: np.array, Y: np.array) -> float:
    if X.size != Y.size:
        print("SIZES MUST MATCH")
        return 0
    else:
        tot_len = 0
        for i in range(X.size - 1):
            dx      = X[i+1] - X[i]
            dy      = Y[i+1] - Y[i]
            length  = math.sqrt(dx*dx + dy*dy)
            tot_len += length
        print("total limb length= ", tot_len)
        return tot_len

def find_first_contact(x_nock: float, y_nock: float, X: np.array, Y: np.array) -> int:
    if X.size != Y.size:
        print("SIZES MUST MATCH")
        return 0
    else:
        cosThetas = np.zeros(X.size)
        sinThetas = np.zeros(Y.size)
        for i in range(X.size):
            dx = X[i] - x_nock
            dy = Y[i] - y_nock
            mag = math.sqrt(dx*dx + dy*dy)
            cosThetas[i] = dx/mag
            sinThetas[i] = dy/mag
        maxcosarg = np.argmax(abs(cosThetas))
        minsinarg = np.argmin(abs(sinThetas))
        print(maxcosarg, minsinarg)
        if maxcosarg != minsinarg:
            print("WHOOPS")
            return 0
        else:
            return maxcosarg    

def build_limb_geometry() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # -- INTERPOLATE PROFILE --
    X = np.array(Spline["X"])
    Y = np.array(Spline["Y"])
    N = int(General["Limb elements"])

    intX, intY = interpolate_2d(X, Y, N)
    print(intX.size)
    print(intY.size)

    # -- INTERPOLATE WIDTH --
    scale = _curved_len(intX, intY)

    L_w = scale*np.array(Width["Length"]) #scaling to get better results
    W   = np.array(Width["Width"])

    intL_w, intW = interpolate_1d(L_w, W, N)

    # -- INTERPOLATE THICKNESS --
    L_h = scale*np.array(HickoryLayer["Length"]) #scaling to get better interpolation results
    H   = np.array(HickoryLayer["Height"])

    intL_h, intH = interpolate_1d(L_h, H, N)

    # -- RESAMPLE SO THEY ARE ALL THE SAME SIZE --
    ysize = intY.size
    wsize = intW.size
    hsize = intH.size
    new_N = min([ysize, wsize, hsize])

    X_unbraced  = np.zeros(new_N)
    Y_unbraced  = np.zeros(new_N)
    Lw_unbraced = np.zeros(new_N)
    W_unbraced  = np.zeros(new_N)
    Lh_unbraced = np.zeros(new_N)
    H_unbraced  = np.zeros(new_N)

    for n in range(new_N):
        ratio = n / new_N
        yn = int(round(ratio * ysize))
        wn = int(round(ratio * wsize))
        hn = int(round(ratio * hsize))

        X_unbraced[n]  = intX[yn]
        Y_unbraced[n]  = intY[yn]

        Lw_unbraced[n] = intL_w[wn]
        W_unbraced[n]  = intW[wn]

        Lh_unbraced[n] = intL_h[hn]
        H_unbraced[n]  = intH[hn]

     
    return X_unbraced, Y_unbraced, Lw_unbraced, W_unbraced, Lh_unbraced, H_unbraced

def build_bracer(x_nock: float, y_nock: float, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if X.size != Y.size:
        print("SIZES MUST MATCH")
        return [x_nock, 0], [y_nock, 0]
    else:
        contact_indx = find_first_contact(x_nock, y_nock, X, Y)

        X_string = X[contact_indx:]
        Y_string = Y[contact_indx:]
        print("bracing strn lens ", X_string.size, Y_string.size)        
        return X_string, Y_string

def build_string(x_nock: float, y_nock: float, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if X.size != Y.size:
        print("SIZES MUST MATCH")
        return [x_nock, 0], [y_nock, 0]
    else:
        contact_indx = find_first_contact(x_nock, y_nock, X, Y)
        N = int(General["String elements"])
        X_string, Y_string = interpolate_2d(np.array([x_nock, X[contact_indx]]), np.array([y_nock, Y[contact_indx]]), N)
        print("strn lens ", X_string.size, Y_string.size)
        #remove last element
        X_string = np.delete(X_string, -1)
        Y_string = np.delete(Y_string, -1)
        contacts = np.zeros(X_string.size)
        print("strn lens ", X_string.size, Y_string.size, contacts.size)

        X_string = np.append(X_string, X[contact_indx:])
        Y_string = np.append(Y_string, Y[contact_indx:])
        true     = np.ones(Y[contact_indx:].size)
        contacts = np.append(contacts, true)
        print("strn lens ", X_string.size, Y_string.size, contacts.size)        
        return X_string, Y_string


def _Ms_limbs(rho: float, W: np.ndarray, H: np.ndarray, L: np.ndarray) -> np.ndarray:
    if W.size != H.size != L.size:
        print("W, H, and L should all be the same size")
        return np.zeros([1,6,6])
    else:
        Ms = np.zeros([L.size, 6, 6])
        for i in range(L.size):
            rho_list    = np.array([rho])
            h_list      = np.array([H[i]])
            Ms[i] = M_limb(rho_list, W[i], h_list, L[i])
        return Ms

def _Kp_limbs(W: np.ndarray, H: np.ndarray, E: float, L: np.ndarray) -> np.ndarray:
    if W.size != L.size != H.shape[0]:
        print("Size of W and L and shape[0] of H all need to be the same")
        return np.zeros([1,6,6])
    else:
        Ks = np.zeros([L.size, 6, 6])
        for i in range(L.size):
            H_list  = np.array([H[i]])
            E_list  = np.array([E])
            Ks[i]   = K_limb(W[i], H_list, E_list, L[i])
        return Ks

def _T_limb(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    if X.size != Y.size:
        print("sizes of X and Y must match")
        return np.zeros([1, 6, 6])
    else:
        T_limb = np.zeros([X.size - 1, 6, 6])
        for i in range(T_limb.shape[0]):
            T = T_g2l_limb(X[i], Y[i], X[i+1], Y[i+1])
            T_limb[i] = T
        return T_limb
        

def build_unbraced_limb() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_unbraced, Y_unbraced, _, W_unbraced, _, H_unbraced = build_limb_geometry()
    print("this needs to be fixed to make multiple layers work")

    #using shape instead of size for W and H because they can be 2d
    if X_unbraced.size != Y_unbraced.size != W_unbraced.size != H_unbraced.shape[0]:
        print("Sizes of X, Y, H, and W need to match")

    print("number of nodes= ", X_unbraced.size)

    L_limb = np.zeros(X_unbraced.size - 1)
    W_limb = np.zeros(W_unbraced.size - 1)
    H_elems = np.zeros(H_unbraced.size - 1)
    for i in range(L_limb.size):
        dx         = X_unbraced[i+1] - X_unbraced[i]
        dy         = Y_unbraced[i+1] - Y_unbraced[i]
        L_limb[i]  = math.sqrt(dx*dx + dy*dy)
        W_limb[i]  = (W_unbraced[i+1] + W_unbraced[i]) / 2
        H_elems[i] = (H_unbraced[i+1] + H_unbraced[i]) / 2
    
    rho = Hickory["Rho"]
    Mp_limb = _Ms_limbs(rho, W_limb, H_elems, L_limb)
    
    print("Number of length elements= ", L_limb.size)
    print("Number of width elements= ", W_limb.size)
    print("Number of height elements= ", H_elems.shape[0])
    print("Number of mass elements= ", Mp_limb.shape[0])

    E = Hickory["E"]
    Kp_limb = _Kp_limbs(W_limb, H_elems, E, L_limb)
    print("Number of local K elements= ", Kp_limb.shape[0])
    
    T_limb_unbraced = _T_limb(X_unbraced, Y_unbraced)
    print("Number of transforms= ", T_limb_unbraced.shape[0])
    
    return X_unbraced, Y_unbraced, Mp_limb, Kp_limb, T_limb_unbraced


if __name__ == "__main__":
    X_unbraced, Y_unbraced, Lw_unbraced, W_unbraced, Lh_unbraced, H_unbraced = build_limb_geometry()

    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    plt.title("Bow limb geometry")
    axs[0].set_title("Profile")
    axs[0].plot(X_unbraced, Y_unbraced, color = "red")
    axs[0].set_ylabel("mm")

    axs[1].set_title("Width")
    axs[1].plot(Lw_unbraced, W_unbraced, color = "red")
    axs[1].set_ylabel("mm")

    axs[2].set_title("Thickness")
    axs[2].plot(Lh_unbraced, H_unbraced, color = "red")
    axs[2].set_xlabel("mm")
    axs[2].set_ylabel("mm")

    plt.show()

    from BowModel.Dimensions import Draw
    x_nock = 0
    y_nock = -Draw["Brace height"]

    print("Nock= ", x_nock, y_nock)

    contact_indx = find_first_contact(x_nock, y_nock, X_unbraced, Y_unbraced)
    print(contact_indx)

    X_bracer, Y_bracer = build_bracer(x_nock, y_nock, X_unbraced, Y_unbraced)
    print("Lens of bracer ", X_bracer.size, Y_bracer.size)

    X_braceV = np.array([x_nock, X_bracer[0]])
    Y_braceV = np.array([y_nock, Y_bracer[0]])

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_title("Contact Points and bracing vector")
    ax.plot(X_unbraced, Y_unbraced, color = "red")
    ax.plot(X_bracer, Y_bracer, color = "blue")
    ax.plot(X_braceV, Y_braceV, color = "green")
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    plt.show()





