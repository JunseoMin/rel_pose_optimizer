import numpy as np
from ..util import *

def makeIdentity(transformations: np.ndarray):
    r"""
    Set the Last object Identity and make the last object reference frame 

    @param 
    input: 
        transformations: np.ndarray type with (number of cameras, number of objects, SE(3))
    output:
        np.ndarray type with (number of cameras, number of objects, SE(3))
    """
    n_camera = transformations.shape[0]
    n_obj = transformations.shape[1]

    updated_transformations = transformations.copy()

    for c in range(n_camera):
        T_clo = updated_transformations[c][-1]   # T_cp  transformation from phone -> action cam (actually, last obj -> curr cam)
        T_loC = inverseSE3(T_clo)
        
        for o in range(n_obj):
            T_co = transformations[c][o]
            updated_transformations[c][o] = T_loC @ T_co

    return updated_transformations
