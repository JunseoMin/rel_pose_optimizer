from .Solver import poseSolver
from .CamRelPoseFromMarker import CamRelPoseFromMarker
from .nanFiller import nanFiller
from .makeIdentity import makeIdentity

from ..util import *

class RelPoseOptimzer():
    
    r"""
    Find optimized reletive pose from **'one'** frame

    ### Input:
        Raw transformation between objects list (n_camera, n_objects, SE(3) matrix (4,4))

    ### Output:
        Optimized transformations (n_objects, SE(3) matrix (4,4))
    """
    def __init__(self, solver_type = "meansolver"):
        self.raw_transformations        = None
        self.camera_transformations     = None
        self.transformations_optimized  = None
        self.solver                     = poseSolver(type=solver_type)

    def setInput(self, transformations):
        r"""
        @param transformations: np.ndarray type with (number of cameras, number of objects, SE(3))
        """
        self.raw_transformations    = transformations

    def optimize(self):
        assert self.raw_transformations is not None, "[Error] Should set input first."
        
        # 0. get camera rel pose from overlapping marker
        self.camera_transformations = CamRelPoseFromMarker(self.raw_transformations)
        assert not (self.camera_transformations == None), "[Error] Camera relative transformation not given."

        # 1. filling na by camera transformations
        self.raw_transformations = nanFiller(self.raw_transformations, self.camera_transformations)

        # 2. make last identity (reference frame)
        # obj_transformations contains each camera's perspective of transformation view
        self.transformations_optimized =  makeIdentity(self.raw_transformations)
        # 3. optimize transformations by solver
        self.transformations_optimized = self.solver.solve(self.transformations_optimized)
        # print(self.transformations_optimized)


    def getRelPose(self):
        assert self.transformations_optimized is not None, "[Error] Should call solver first"
        return self.transformations_optimized

    def clear(self):
        self.raw_transformations        = None
        self.camera_transformations     = None
        self.transformations_optimized  = None        
    