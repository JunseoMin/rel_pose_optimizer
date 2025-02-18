import numpy as np

def inverseSE3(T: np.ndarray):
    assert T.shape == (4,4), "input shape miss match!! check inverse function"

    R = T[:3,:3]
    t = T[:3,3]

    R_inv = R.T
    t_inv = -R_inv @ t

    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv