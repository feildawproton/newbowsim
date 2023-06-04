#trying to implement the newton-raphson method described in VirtualBow
import numpy as np
np.set_printoptions(linewidth=np.inf)
from typing import Tuple

from build_bow import build_unbraced_limb
from assembly import calc_T_g2l_limb, Ms_l2g, insert_gKlimb, update_pos

from jax.scipy.optimize import minimize
import jax.numpy as jnp

import matplotlib.pyplot as plt

from BowModel.Dimensions import Draw

def _dispctrl_c_i(u_i: np.ndarray, ctrl_ndc: np.ndarray, ctrl_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert ctrl_ndc.size == ctrl_val.size
    assert np.max(ctrl_ndc) < u_i.size
    
    #the control vector starts as a copy of the u_i
    #then insert the control values in ubar at the correct location describes by ctrl_ndc
    #the partial derivative of c with respect to u is simply an indicator of whick values are controlled
    #where 1 indicates the value is controlled and 0 indicates it is free
    #the partial derivative of c with repect to lambda is all 0's
    ubar    = np.copy(u_i)
    dcOdu   = np.zeros(ubar.size)
    for i, ndx in enumerate(ctrl_ndc):
        ubar[ndx]   = ctrl_val[i]
        dcOdu[ndx]  = 1
        
    c_i = np.subtract(u_i, ubar)
    
    dcOdlam = np.zeros(ubar.size)
    
    return c_i, dcOdu, dcOdlam

#this function calculates the load error for a simulation that is load controlled
#this is simply contraining the external forces to mostly be zero
#except where displacement is controlled
#dc over dlam will mostly be ones
def _c_lam_uctrl(lam_i: np.ndarray, u_ctrl_ndc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pbar    = np.zeros(lam_i.size)
    dcOdlam = np.ones(lam_i.size)
    for ndx in u_ctrl_ndc:
        pbar[ndx]       = lam_i[ndx]
        dcOdlam[ndx]    = 0
        
    lam_i = np.subtract(lam_i, pbar)
    
    dcOdu = np.zeros(lam_i.size)
    
    return lam_i, dcOdu, dcOdlam
    

def _alpha_i(lam_i: np.ndarray, K_0: np.ndarray, K_i: np.ndarray, u_i: np.ndarray) -> np.ndarray:
    '''
    p_i     = np.matmul(K_i, u_i)
    lam_1m1 = np.add(lam_i, -1)
    result  = np.multiply(p_i, lam_1m1)
    
    alpha_i = np.linalg.solve(K_i, result)
    '''
    lam_1m1 = np.add(lam_i, -1)
    alpha_i = np.multiply(u_i, lam_1m1)
    return alpha_i

def _beta_i(K_0: np.ndarray, u_i: np.ndarray, K_i: np.ndarray) -> np.ndarray:
    result = np.matmul(K_i, u_i)
    print(result)
    beta_i = np.linalg.solve(K_i, result)
    return beta_i

def _DELTAlam_i(c_i: np.ndarray, 
                dcOdu_i: np.ndarray, dcOdlam_i: np.ndarray, 
                alpha_i: np.ndarray, beta_i: np.ndarray) -> np.ndarray:
    dcOduAlpha_i    = np.multiply(dcOdu_i, alpha_i)
    dcOduBeta_i     = np.multiply(dcOdu_i, beta_i)
    negc_i          = np.multiply(-1, c_i)
    
    numerator_i     = np.add(negc_i, dcOduAlpha_i)
    print("dcOduBeta_i")
    print(dcOduBeta_i)
    print("dcOdlam_i")
    print(dcOdlam_i)
    denominator_i   = np.add(dcOduBeta_i, dcOdlam_i)
    
    DELTAlam_i      = np.divide(numerator_i, denominator_i)
    
    return DELTAlam_i

if __name__ == "__main__":
    X_unbraced, Y_unbraced, Mp_limb, Kp_limb, T_limb_unbraced = build_unbraced_limb()
    
    #test disp control
    #need to use postive valued ndcs for the contraint
    #otherwise the error calculation will skip it
    num_nodes   = X_unbraced.size
    u_ndc       = np.array([0, 1, 2, 3*num_nodes-2], dtype=int)
    u_ctrl      = np.array([0, 0, 0, -1], dtype=float)
    
    #generate global K_0
    #not strickly necessary but we will init u_0 for use later
    u_0         = np.zeros(3 * num_nodes)
    X, Y        = update_pos(X_unbraced, Y_unbraced, u_0)
    Ts_limb     = calc_T_g2l_limb(X, Y)
    gKs_limb    = Ms_l2g(Kp_limb, Ts_limb)
    gK_0        = np.zeros([3*num_nodes, 3*num_nodes])
    insert_gKlimb(gK_0, gKs_limb)

    #generate global K_i
    #from the guess u_i
    u_i         = np.multiply( 0.1, np.ones(3 * num_nodes))
    u_i[0]      = 0.
    u_i[1]      = 0.
    u_i[2]      = 0.
    
    X_i, Y_i    = update_pos(X, Y, u_i)
    Ts_limb_i   = calc_T_g2l_limb(X_i, Y_i)
    gKs_limb_i  = Ms_l2g(Kp_limb, Ts_limb_i)
    gK_i        = np.zeros([3*num_nodes, 3*num_nodes])
    insert_gKlimb(gK_i, gKs_limb_i)

    u_c_i, u_dcOdu_i, u_dcOdlam_i   = _dispctrl_c_i(u_i, u_ndc, u_ctrl)
    lam_i = np.matmul(gK_i, u_i)
    lam_c_i, p_dcOdu_i, p_dcOdlam_i = _c_lam_uctrl(lam_i, u_ndc)
    c_i         = np.add(u_c_i, lam_c_i)
    dcOdu_i     = np.add(u_dcOdu_i, p_dcOdu_i)
    dcOdlam_i   = np.add(u_dcOdlam_i, p_dcOdlam_i)
    
    print("c_i")
    print(c_i)
    print("dcOdu_i")
    print(dcOdu_i)
    print("dcOdlam_i")
    print(dcOdlam_i)

    
    print("i'm not sure if this is right but i'm initializig lam_i to ones")
    lam_i = np.zeros(3 * num_nodes)
    print("lam_i")
    print(lam_i)
    alpha_i = _alpha_i(lam_i, gK_0, gK_i, u_i)
    print("alpha_i")
    print(alpha_i)
    beta_i = u_i#_beta_i(gK_0, u_i, gK_i)
    print("beta_i")
    print(beta_i)

    DELTAlam_i = _DELTAlam_i(c_i, dcOdu_i, dcOdlam_i, alpha_i, beta_i)
    print("DELTAlam_i")
    print(DELTAlam_i)

    delta_u_i = alpha_i + beta_i * DELTAlam_i
    print("delta u_i")
    print(delta_u_i)
    
    

    