# rel_pose_optimizer
calc relative pose from n_cameras with n_objects

**Input** - (m, 4, 4) => m object, SE3(4,4) transformation matrix  
**Output** - (l, 4, 4) => l optimal pose, SE3(4,4) transformation matrix-- optimized

## Install
```
cd /path/to/your/project
git clone https://github.com/JunseoMin/rel_pose_optimizer.git
```

## Example code
The directory should be:
```
.
├── example.py
└── rel_pose_optimizer

```

### Example without visualization
```Python3
from rel_pose_optimizer.PoseOptimizer import RelPoseOptimzer
from rel_pose_optimizer.util import is_SE3

import numpy as np

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

data  = np.load("/path/to/rel_pose_optimizer/datset/result_action5.npy")
data2 = np.load("/path/to/rel_pose_optimizer/datset/result_12promax.npy")
data3 = np.load("/path/to/rel_pose_optimizer/datset/result_15navy.npy")

new_data = np.stack([data, data2, data3], axis=1)

optimizer = RelPoseOptimzer("meansolver")

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
```


### Visualization Example
```Python3
from rel_pose_optimizer.PoseOptimizer import RelPoseOptimzer
from rel_pose_optimizer.util import is_SE3, visualize_pose
import numpy as np


passed = 0

data  = np.load("/home/junseo/sejun/rel_pose_optimizer/datset/result_action5.npy")
data2 = np.load("/home/junseo/sejun/rel_pose_optimizer/datset/result_12promax.npy")
data3 = np.load("/home/junseo/sejun/rel_pose_optimizer/datset/result_15navy.npy")

new_data = np.stack([data, data2, data3], axis=1)

optimizer = RelPoseOptimzer("meansolver")

optimizer.setInput(data)
optimizer.optimize()
relpose = optimizer.getRelPose()

visualize_pose(new_data[0][0])
visualize_pose(relpose)
```