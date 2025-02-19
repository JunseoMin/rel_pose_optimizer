import numpy as np
from einops import rearrange
from .util import inverseSE3, is_SE3, visualize_pose
from .PoseOptimizer.RelPoseOptimizer import RelPoseOptimzer


def test_optimizer(data, optimizer: RelPoseOptimzer):

    optimizer.setInput(data)
    optimizer.optimize()
    relpose = optimizer.getRelPose()
    optimizer.clear()

    for transformation in relpose:
        valid,e = is_SE3(transformation)
        assert valid, e

    return 1

passed = 0


# data  = np.load("path/to/rel_pose_optimizer/datset/result_action5.npy")
# data2 = np.load("path/to/rel_pose_optimizer/datset/result_12promax.npy")
# data3 = np.load("path/to/rel_pose_optimizer/datset/result_15navy.npy")

new_data = np.load("/home/junseo/sejun/rel_pose_optimizer/datset/combined.npy")

d = new_data[0]
optimizer = RelPoseOptimzer("meansolver")
optimizer.setInput(d)
optimizer.optimize()
relpose = optimizer.getRelPose()

visualize_pose(relpose)
print(new_data[0][0])

errors = []
for i,data in enumerate(new_data):
    try:
        passed += test_optimizer(data, optimizer)
    except Exception as e:
        errors.append(i)
        if e == "test fail":
            print("error!!!!")
        

print(f"passed {passed} / {len(new_data)}")
print("error frames:")
print(errors)