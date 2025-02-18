import numpy as np
from einops import rearrange
from util import inverseSE3
from PoseOptimizer import *

data  = np.load("/home/junseo/sejun/rel_pose_optimizer/datset/result_action5.npy")
data2 = np.load("/home/junseo/sejun/rel_pose_optimizer/datset/result_12promax.npy")
data3 = np.load("/home/junseo/sejun/rel_pose_optimizer/datset/result_15navy.npy")

new_data = np.stack([data, data2, data3], axis=1)


def test_CamRelPoseFromMarker(data):
    transformation = data

    print("Input transformation shape:", transformation.shape) 

    rel_poses = CamRelPoseFromMarker(transformation)

    print("Output shape:", rel_poses.shape)

    print(rel_poses)
    assert rel_poses.shape[0] == 1, "Only one object's transformation should be used due to break statement"
    assert rel_poses.shape[1] == 3, "3 relative transformations expected (c1c2, c1c3, c2c3)"
    assert rel_poses.shape[2:] == (4, 4), "Each transformation must be a 4x4 matrix"

    assert not np.any(np.isnan(rel_poses)), "NaN values should be ignored"

    print("Test passed! Function works as expected.")

    return 1

passed = 0

for data in new_data[120:]:
    passed += test_CamRelPoseFromMarker(data)
    print(f"passed {passed} / {len(new_data[120:])}")
