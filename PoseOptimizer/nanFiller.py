from typing import Tuple
import numpy as np
from einops import rearrange
from util.math import *
from util.error import *

def nanFiller(obj_transformations: np.ndarray, cam_transformations: dict):
    r"""
    ## Brief:
    Fill object transformations from camera relative positions. Expects that cam_transformations contains (T_12, T_13, T_23 ...)

    @param obj_transformations: Object transformations with shape (n_camera, n_objects, SE(3) matrix(4,4)).
    @param cam_transformations: Camera transformations with shape ().

    ## Return:
    Processed transformations with NaN values filled.
    """

    n_camera    = obj_transformations.shape[0]
    n_obj       = obj_transformations.shape[1]

    for c1 in range(n_camera):
        for o in range(n_obj):  
            # transformation matrix o'th object to c1'th camera
            T_c1o = obj_transformations[c1][o]

            nan_cnt = 0
            if np.any(np.isnan(T_c1o)): # Transformation matrix contains nan
                for c2 in range(n_camera):
                    if c1 == c2:
                        continue

                    T_c2o = obj_transformations[c2][o]

                    if np.any(np.isnan(T_c2o)):
                        # Skip if other camera doesn't contains the Transformation
                        nan_cnt += 1
                        assert (nan_cnt != n_camera - 1) , "[ERROR] one of the object doesn't observed from whole camera"
                        continue
                    
                    # T_c1o = Tc1c2 Tc2o
                    if f"T{c1}{c2}" in cam_transformations:
                        T_c1c2 = cam_transformations[f"T{c1}{c2}"]
                    elif f"T{c2}{c1}" in cam_transformations:
                        T_c1c2 = inverseSE3(cam_transformations[f"T{c2}{c1}"])
                    else:
                        raise LogicalError(f"[ERROR] No valid transformation found for T_{c1}{c2} or T_{c2}{c1}")

                    T_c1o = T_c1c2 @ T_c2o
                    obj_transformations[c1][o] = T_c1o

    return obj_transformations