import numpy as np

data = np.load("/home/junseo/sejun/rel_pose_optimizer/datset/tags_12promax.npy", allow_pickle=True)

for dt in data:
    print(dt)