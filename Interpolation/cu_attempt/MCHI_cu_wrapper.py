import ctypes
from ctypes import *
import numpy as np

def get_interpolate_Ts():
    dll = ctypes.CDLL("MCHI_funcs.so", mode=ctypes.RTLD_GLOBAL)
    func = dll.interpolate_Ts
    func.argtypes = [c_int, POINTER(c_float),
                     c_int, POINTER(c_float), POINTER(c_int)]
    return func

def cu_interpolate_Ts(Ts, T_interpolated, Ts_k_indices):
    N_Ts    = Ts.size
    N       = T_interpolated.size
    
    # convert to ctypes
    #cN_Ts           = N_Ts.ctypes.data_as(c_int)
    pTs             = Ts.ctypes.data_as(POINTER(c_float))
    #cN              = N.ctypes.data_as(c_int)
    pT_interpolated = T_interpolated.ctypes.data_as(POINTER(c_float))
    pTs_k_indices   = Ts_k_indices.ctypes.data_as(POINTER(c_int))
    
    # call func
    interpolate_TS = get_interpolate_Ts()
    interpolate_TS(N_Ts, pTs, N, pT_interpolated, pTs_k_indices)
    
    # convert back to numpy
    T_interpolated = np.fromiter(pT_interpolated, dtype=np.float32, count=N)
    Ts_k_indices = np.fromiter(pTs_k_indices, dtype=np.int32, count=N)
    
    