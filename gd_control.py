import numpy as np
np.set_printoptions(linewidth=np.inf)

from build_bow import build_unbraced_limb
from assembly import calc_T_g2l_limb, Ms_l2g, insert_gKlimb, update_pos

def _dfu_du_i(u_i: np.ndarray, u_ctrl_i: np.ndarray, ctrl_uval_i: np.ndarray) -> np.ndarray:
    assert u_ctrl_i.size == ctrl_uval_i.size
    
    #initial ubar to be equal to u_i
    #except where we have a control value
    #as indicated by the control index
    ubar = np.copy(u_i)
    for c, n in enumerate(u_ctrl_i):
        ubar[n] = ctrl_uval_i[c]
        
    df_du = np.subtract(u_i, ubar)
    scale = 1 / df_du.size  #1/N
    df_du = np.multiply(scale, df_du)
    
    return df_du

def _dfp_dp_i(p_i: np.ndarray, u_ctrl_i: np.ndarray) -> np.ndarray:
    assert u_ctrl_i.size == ctrl_uval_i.size
    
    #initialize pbar as zeros
    #except for where the node is in the u controlled list
    #then set it to p_i so that it is free
    pbar = np.zeros(p_i.size)
    for c in u_ctrl_i:
        pbar[c] = p_i[c]
        
    dfu_dp  = np.subtract(p_i, pbar)
    scale   = 1 / dfu_dp.size #1/M
    dfu_dp  = np.multiply(scale, dfu_dp) 
    
    return dfu_dp

def _dp_du(K_i: np.ndarray) -> np.ndarray:
    assert K_i.shape[0] == K_i.shape[1]
    
    dp_du = np.transpose(K_i)
    
    return dp_du

if __name__ == "__main__":
    X_unbraced, Y_unbraced, Mp_limb, Kp_limb, T_limb_unbraced = build_unbraced_limb()
    
    #test gd combining displacement and 
    num_nodes   = X_unbraced.size
    u_ctrl_i    = np.array([0, 1, 2, -2], dtype=int)
    ctrl_uval_i = np.array([0, 0, 0, -1], dtype=float)
    
    u_i = np.zeros(num_nodes * 3)
    dfu_du = _dfu_du_i(u_i, u_ctrl_i, ctrl_uval_i)
    print("dfu_du")
    print(dfu_du)
    
    p_i = np.ones(num_nodes * 3)
    dfp_dp = _dfp_dp_i(p_i, u_ctrl_i)
    print("dfp_dp")
    print(dfp_dp)
    
    #generate global K_0
    Ts_limb     = calc_T_g2l_limb(X_unbraced, Y_unbraced)
    gKs_limb    = Ms_l2g(Kp_limb, Ts_limb)
    gK_0        = np.zeros([3*num_nodes, 3*num_nodes])
    insert_gKlimb(gK_0, gKs_limb)
    
    #print("K_i")
    #print(gK_0)
    
    dp_du = _dp_du(gK_0)
    print("dp_du")
    print(dp_du)
    
    dfp_du = np.matmul(dp_du, dfp_dp)
    
    print("dfp_du")
    print(dfp_du)
    
    dfp_du = np.matmul(dfp_dp, dp_du)
    
    print("dfp_du")
    print(dfp_du)