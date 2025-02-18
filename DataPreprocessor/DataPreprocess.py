import numpy as np
from einops import rearrange

data  = np.load("/home/junseo/sejun/rel_pose_optimizer/datset/result_action5.npy")
data2 = np.load("/home/junseo/sejun/rel_pose_optimizer/datset/result_12promax.npy")
data3 = np.load("/home/junseo/sejun/rel_pose_optimizer/datset/result_15navy.npy")

assert data.shape == data2.shape, "Input shape should be the same"
assert data2.shape == data3.shape, "Input shape should be the same"

print("Original shape:", data.shape)  # (num_frames, num_objects, 4, 4)

num_frames, num_objects, _, _ = data.shape
num_cameras = 3  # (action5, 12promax, 15navy)

for da, dp, dn in zip(data, data2, data3):
    assert da.shape[0] == dp.shape[0] == dn.shape[0], "Number of objects must be the same in each dataset"

# (num_frames, num_cameras, num_objects, 4, 4)
new_data = np.stack([data, data2, data3], axis=1)

print("Transformed shape:", new_data.shape)
# np.save("/home/junseo/sejun/rel_pose_optimizer/datset/combined.npy",new_data)
print(new_data[0])    # ex) 0frame 0camera 0 obj

print("------------------------------------------------")
newT = rearrange(new_data[0], "c o row col -> o c row col")
print(newT)    # ex) 0frame 0camera 0 obj