import numpy as np
from util import inverseSE3
from PoseOptimizer import *

def test_makeIdentity(data):
    n_camera,n_objects,_,_ = data.shape

    cam_transformations = CamRelPoseFromMarker(data)

    filled_transformations = nanFiller(data, cam_transformations)

    updated_transformations = makeIdentity(filled_transformations)

    for c in range(n_camera):
        np.testing.assert_array_almost_equal(updated_transformations[c, -1], np.eye(4), decimal=6)

    for c in range(n_camera):
        for o in range(n_objects):
            R = updated_transformations[c, o][:3, :3]
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-6), "Rotation matrix should be orthogonal"
            assert np.isclose(np.linalg.det(R), 1, atol=1e-6), "Rotation matrix determinant should be 1"

    return 1

passed = 0

data  = np.load("/home/junseo/sejun/rel_pose_optimizer/datset/result_action5.npy")
data2 = np.load("/home/junseo/sejun/rel_pose_optimizer/datset/result_12promax.npy")
data3 = np.load("/home/junseo/sejun/rel_pose_optimizer/datset/result_15navy.npy")

new_data = np.stack([data, data2, data3], axis=1)

for data in new_data:
    try:
        passed += test_makeIdentity(data)
    except Exception as e:
        if e == "Rotation matrix should be orthogonal" or e == "Rotation matrix determinant should be 1":
            print("nan fill test failed!")

print(f"passed {passed} / {len(new_data)}")

