from typing import Tuple
import numpy as np
from einops import rearrange

def nanFiller(obj_transformations: np.ndarray, cam_transformations: np.ndarray):
    r"""
    ## Brief:
    Fill object transformations from camera relative positions. Expects that cam_transformations contains (T_12, T_13, T_23 ...)

    @param obj_transformations: Object transformations with shape (n_camera, n_objects, SE(3) matrix(4,4)).
    @param cam_transformations: Camera transformations with shape ().

    ## Return:
    Processed transformations with NaN values filled.
    """
    