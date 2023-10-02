import ctypes
import numpy as np

# -- wrapper around MCHI implementation -- #
# implemented as a class so that we only need to load the lib when we inst the lib
# should this be a static var so that we only need to do that once?
class Interpolator:
    what = "wrapper over omp implementation of MCHI"  # a class var (shared across all classes)
    
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
        #                T_k_ndc                , T_intrpltd             , 
        #                Y_intrpltd
        #
        # with constness const                  , const                  , 
        #                const                  , const                  , const                  ,
        #                const                  , const                  ,
        #                mutates
        #
        func.argtypes = [ctypes.c_uint          , ctypes.c_uint          , 
                         ctypes.pointer(c_float), ctypes.pointer(c_float), ctypes.pointer(c_float),
                         ctypes.pointer(c_uint) , ctypes.pointer(c_float), 
                         ctypes.pointer(c_float)]
        return func

    def   __init__(self):
        #self.method=method # prolly could use this
        self.lib        = self.__get_lib()
        self.intrplte_T = self.__get_intrplte_T(lib=self.lib)
        self.calc_M     = self.__get_calc_M(lib=self.lib)
        self.intrplte_Y = self.__get_intrplte_Y(lib=self.lib)

    def interpolate_T(self, T: np.ndarry, T_intrpltd: np.ndarray, T_k_ndc: np.ndarray):
        N_orig        = T.size[0]
        N_new         = T_intrpltd.size[0]
        
        assert N_new >= N_orig
        assert N_new == T_k_ndc.size[0]

        pT            =          T.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pT_intrpltd   = T_intrpltd.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pT_k_ndc      =    T_k_ndc.ctypes.data_as(ctypes.POINTER(c_uint))

        self.intrplte_T(N_orig, N_new, pT, pT_intrpltd, pT_k_ndc)

        T_intrpltd    = np.fromiter(pT_intrpltd, dtype=np.float32, count=N_new)
        T_k_ndc       = np.fromiter(T_k_ndc    , dtype=np.uint32 , count=N_new)

    def calc_M(self, T: np.ndarray, Y: np.ndarray, M: np.ndarray):
        N  = T.size[0]

        assert N == Y.size[0] ==  M.size[0]

        pT = T.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pY = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pM = M.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.calc_M(N, pT, pY, pM)

        M  = np.fromiter(pM, dtype=np.float32, count=N)

    def interpolate_Y(self,
                      T: np.ndarray         , Y: np.ndarray         , M: np.ndarray, 
                      T_k_ndc: np.ndarray   , T_intrpltd: np.ndarray, 
                      Y_intrpltd: np.ndarray):

        N_orig      = T.size[0]
        N_new       = T_k_ndc[0]
        assert N_orig == Y.size[0] == M.size[0]
        assert N_new  == T_intrpltd.size[0] == Y_intrpltd.size[0]

        pT          =          T.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pY          =          Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pM          =          M.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pT_k_ndc    =    T_k_ndx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
        pT_intrpltd = T_intrpltd.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pY_intrpltd = Y_intrpltd.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.interpolate_Y(N_orig     , N_new      ,
                           pT         , pY         , pM,
                           pT_k_ndc   , pT_intrpltd, 
                           pY_intrpltd)

def interpolated_Ys(self, T: np.ndarray, Ys: np.ndarray, N: int
                   ) -> Tuple[np.ndarray, np.ndarray]:
    N_orig = T.shape[0]
    #N_feat = Ys.shape[-1]
    assert N_orig == Ys.shape.[0]
    assert len(T.shape) < 2
    assert lent(Ys.shape) < 3
    assert N_orig <= N

    T_intrpltd  = np.zeros(N)
    T_k_indices = np.zeros(N, dtype.np.uint32)

    self.interpolate_T(T, T_intrpltd, T_k_ndc)

    # expand dims if needed
    if len(Y.shape) < 2:
        Ys = np.expand(Ys, axis = -1)
    
    N_feats    = Ys.shape[-1]
    Y_intrpltd = np.zeros(N, N_feats)
    for feat in range(N_feats):
        M_feat = self.calc_M(T, Ys[:, feat])
        self.interpolate_Y(T, Ys[:, feat], M_feat, T_k_ndx, T_intrpltd, Y_intrpltd)

    # collapse in the case of N_feats = 1
    if N_feats == 1:
        Y_intrpltd = np.reduce(Y_intrpltd, axis=-1)

    return T_intrpltd, Y_intrpltd

