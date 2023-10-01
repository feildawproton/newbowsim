import ctypes
import numpy as np

class Interpolator:
    what = "wrapper some omp implementation of interpolation"  # a class var (shared across all classes)
    
    def __get_lib(path="MCHI.so"):
        lib = ctypes.CDLL(path)
        return lib
        
    def __get_intrplte_T(lib: ctypes.CDLL):
        func          = lib.interpolate_T_omp
        # these are      N_orig    ,              N_new,         
        #                T         ,                       
        #                T_intrpltd,              T_k_ndc
        #
        # with constness const     ,              const,
        #                const     ,                   
        #                mutates   ,              mutates
        func.argtypes = [ctypes.c_uint,           ctypes.c_uint, 
                         ctypes.POINTER(c_float), 
                         ctypes.POINTER(c_float), ctypes.POINTER(c_uint)]
        return func

    def __get_calc_M(lib: ctypes.CDLL):
        func          = lib.calc_M
        # these are      N            , X or T                 , Y                      , M
        # with constness const        , const                  , const                  , mutates
        func.argtypes = [ctypes.c_uint, ctypes.POINTER(c_float), ctypes.POINTER(c_float), ctypes.POINTER(c_float)]
        return func

    def __get_intrplte_Y(lib: ctypes.CDLL):
        func          = lib.interpolate_Y_omp
        # these are      N_original             , N_new                  ,
        #                T                      , Y                      , M                      ,
        #                T_k_ndc                ,
        #                T_intrpltd             , Y_intrpltd
        #
        # with constness const                  , const                  , 
        #                const                  , const                  , const                  ,
        #                const                  , 
        #                mutates                , mutates
        #
        func.argtypes = [ctypes.c_uint          , ctypes.c_uint          , 
                         ctypes.pointer(c_float), ctypes.pointer(c_float), ctypes.pointer(c_float),
                         ctypes.pointer(c_uint) ,
                         ctypes.pointer(c_float), ctypes.pointer(c_float)]
        return func

    def   __init__(self):
        #self.method=method # prolly could use this
        self.lib        = self.__get_lib()
        self.intrplte_T = self.__get_intrplte_T(lib=self.lib)
        self.calc_M     = self.__get_calc_M(lib=self.lib)
        self.intrplte_Y = self.__get_intrplte_Y(lib=self.lib)

    def interpolate_T(T: np.ndarry, T_intrpltd: np.ndarray, T_k_ndc: np.ndarray):
        N_orig        = T.size[0]
        N_new         = T_intrpltd.size[0]
        assert N_new >= N_orig
        assert N_new == T_k_ndc.size[0]
        pT            = T.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pT_intrpltd   = T_intrpltd.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pT_k_ndc      = T_k_ndc.ctypes.data_as(ctypes.POINTER(c_uint))

        self.intrplte_T(N_orig, N_new, pT, pT_intrpltd, pT_k_ndc)

        T_intrpltd    = np.fromiter(pT_intrpltd, dtype=np.float32, count=N_new)
        T_k_ndc       = np.fromiter(T_k_ndc    , dtype=np.uint32 , count=N_new)

    def calc_M(T: np.ndarray, Y: np.ndarray, M: np.ndarray):
        N  = T.size[0]
        assert N == Y.size[0] ==  M.size[0]

        pT = T.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pY = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pM = M.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.calc_M(N, pT, pY, pM)

        M  = np.fromiter(pM, dtype=np.float32, count=N)

    def interpolate_Y():



