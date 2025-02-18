from PoseOptimizer.Solver import Solver
from util import *
from PoseOptimizer.nanFiller import *
from PoseOptimizer.CamRelPoseFromMarker import *

class RelPoseOptimzer():
    
    r"""
    @brief ### Find optimized reletive pose from 'one' frame

    ### Input:
        Raw transformation between objects list (n_camera, n_objects, SE(3) matrix (4,4))

    ### Output:
        Optimized transformations (n_objects, SE(3) matrix (4,4))
    """
    def __init__(self,):
        self.raw_transformations        = None
        self.camera_transformations     = None
        self.transformations_optimized  = None

        self.solver                     = Solver()

    def set_input(self, transformations):
        r"""
        @param transformations: np.ndarray type with (number of cameras, number of objects, SE(3))
        """
        self.raw_transformations    = transformations

    def optimize(self):
        assert not (self.raw_transformations == None), "[Error] Should set input first."
        
        # 0. get camera rel pose from overlapping marker
        self.camera_transformations = CamRelPoseFromMarker(self.raw_transformations)
        assert not (self.camera_transformations == None), "[Error] Camera relative transformation not given."

        # 1. filling na by camera transformations
        self.raw_transformations = nanFiller(self.raw_transformations)

        # 2. make last identity (reference frame)
        self.raw_transformations = make_identity(self.raw_transformations)

        # 3. optimize transformations by solver
        self._solve()

    def get_rel_pose(self):
        assert not (self.transformations_optimized == None), "[Error] Should call solver first"
        return self.transformations_optimized

    
    def _solve(self):
        assert not (self.raw_transformations == None), "[Error] Should set input first."
        self.solver.solve()

        pass
    