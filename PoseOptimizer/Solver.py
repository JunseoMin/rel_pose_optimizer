import numpy as np
from einops import rearrange
from ..util.math import averageSE3

import pyceres

class poseSolver():
    def __init__(self, type = "meansolver"):
        assert type in ["meansolver","reproject"], "solver type should be 'meansolver' or 'reproject' "
        
        self.type = type

    def mean_solve(self, transformations:np.ndarray):
        r"""
        solves transformation matrixes mean
        ### input: 
            Object transformations with shape (n_camera, n_objects, SE(3) matrix(4,4)).
        ### output:
            Object transformations with shape (n_objects, SE(3) matrix(4,4)).
        """

        n_camera, n_obj, _, _ = transformations.shape
        transformations = rearrange(transformations, "c o r col -> o c r col")
        
        res = []
        for i in range(n_obj):
            avg = averageSE3(transformations[i])
            res.append(avg)

        return np.array(res)

    def reproject_solve(self, transformation:np.ndarray):
        
        pass

    def solve(self, transformation):
        if self.type == "meansolver":
            res = self.mean_solve(transformation)
            return res
        if self.type == "reproject":
            #TODO
            pass