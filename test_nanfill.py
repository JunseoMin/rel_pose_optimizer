import numpy as np
from PoseOptimizer import nanFiller
from PoseOptimizer import CamRelPoseFromMarker

def test_nanFiller(data):

    cam_transformations = CamRelPoseFromMarker(data)

    # print("Before filling NaNs:", data)
    filled_transformations = nanFiller(data, cam_transformations)

    # print("After filling NaNs:", filled_transformations)

    assert not np.any(np.isnan(filled_transformations)), "There should be no NaN values after processing!"

    return 1

passed = 0

data  = np.load("path/to/rel_pose_optimizer/datset/result_action5.npy")
data2 = np.load("path/to/rel_pose_optimizer/datset/result_12promax.npy")
data3 = np.load("path/to/rel_pose_optimizer/datset/result_15navy.npy")

new_data = np.stack([data, data2, data3], axis=1)

error_frames = []

for i,data in enumerate(new_data):
    try:
        passed += test_nanFiller(data)
    except Exception as e:
        error_frames.append(i)
        pass

print(f"passed {passed} / {len(new_data)}")
print(error_frames)
