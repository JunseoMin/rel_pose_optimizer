import numpy as np
from einops import rearrange
from util import inverseSE3
from PoseOptimizer import *



def test_CamRelPoseFromMarker(data):
    transformation = data

    print("Input transformation shape:", transformation.shape) 

    rel_poses = CamRelPoseFromMarker(transformation)

    # print("Output length:", len(rel_poses))  
    # print("Example output:", rel_poses)  

    assert isinstance(rel_poses, dict), "Each element should be a dictionary"
    # assert all("T01" in obj and "T02" in obj and "T12" in obj for obj in rel_poses), "Keys should be 'T12', 'T13', 'T23'"
    assert not np.any(np.isnan(rel_poses["T12"])), "NaN values should be ignored"

    print("Test passed! Function works as expected.")

    return 1

passed = 0

data  = np.load("path/to/rel_pose_optimizer/datset/result_action5.npy")
data2 = np.load("path/to/rel_pose_optimizer/datset/result_12promax.npy")
data3 = np.load("path/to/rel_pose_optimizer/datset/result_15navy.npy")

new_data = np.stack([data, data2, data3], axis=1)


for data in new_data:
    try:
        passed += test_CamRelPoseFromMarker(data)
    except Exception as e:
        pass

    print(f"passed {passed} / {len(new_data)}")
