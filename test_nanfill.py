import numpy as np
from PoseOptimizer import nanFiller
from PoseOptimizer import CamRelPoseFromMarker

def test_nanFiller(data):

    cam_transformations = CamRelPoseFromMarker(data)

    filled_transformations = nanFiller(data, cam_transformations)

    print("Before filling NaNs:", data)
    print("After filling NaNs:", filled_transformations)

    assert not np.any(np.isnan(filled_transformations)), "There should be no NaN values after processing!"

    print("Test passed! NaN values filled successfully.")

    return 1

passed = 0

data  = np.load("/home/junseo/sejun/rel_pose_optimizer/datset/result_action5.npy")
data2 = np.load("/home/junseo/sejun/rel_pose_optimizer/datset/result_12promax.npy")
data3 = np.load("/home/junseo/sejun/rel_pose_optimizer/datset/result_15navy.npy")

new_data = np.stack([data, data2, data3], axis=1)


for data in new_data[200:]:
    passed += test_nanFiller(data)
    print(f"passed {passed} / {len(new_data[200:])}")
