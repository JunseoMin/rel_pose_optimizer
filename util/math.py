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

def is_SE3(matrix):
    if matrix.shape != (4, 4):  
        return False, "Matrix is not 4x4"

    R = matrix[:3, :3]  
    t = matrix[:3, 3]   
    bottom_row = matrix[3, :]  
    
    identity = np.eye(3)
    if not np.allclose(R.T @ R, identity, atol=1e-6):
        return False, "Rotation part is not orthogonal"

    if not np.isclose(np.linalg.det(R), 1, atol=1e-6):
        return False, f"Determinant of rotation part is not 1 (det(R) = {np.linalg.det(R)})"

    if not np.allclose(bottom_row, [0, 0, 0, 1], atol=1e-6):
        return False, "Last row is not [0, 0, 0, 1]"

    return True, "Matrix is a valid SE(3) transformation"
