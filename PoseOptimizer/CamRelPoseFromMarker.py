import numpy as np
from einops import rearrange
from util import *
from util.error import LogicalError

def CamRelPoseFromMarker(transformation:np.ndarray):
    r"""
    @brief Calc camera relative transformation from overlapping AR marker
    @param transformations: np.ndarray type with (number of cameras, number of objects, SE(3))

    return
        {"T12": Tc1c2,"T13": T_c1c3,"T23": Tc2c3 ...}
    """

    camera_rels = {}

    n_camera    = transformation.shape[0]
    n_objects   = transformation.shape[1]
    newT        = rearrange(transformation, "c o row col -> o c row col") # rearrange to (number of objects, number of cameras, SE(3))

    for i in range(n_objects):
        if len(camera_rels):
            # If relative position extracted
            # Add another option if needed
            break

        if np.any(np.isnan(newT[i])):   # pass if i'th object observation contains NaN
            continue
        
        object_rels = {}

        for c1 in range(n_camera):
            for c2 in range(c1 + 1, n_camera):
                Tc1c2 = newT[i][c1] @ inverseSE3(newT[i][c2])
                object_rels[f"T{c1}{c2}"] = Tc1c2
                
        camera_rels = object_rels

    assert len(camera_rels) > 0, "[ERROR] Can't compute camera relative pose"
    
    return camera_rels