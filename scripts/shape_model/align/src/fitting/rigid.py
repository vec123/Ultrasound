import numpy as np

def match_landmarks(model_names, model_indices, target_dict):
    """
    Returns matched arrays (source, target)
    """

    src = []
    tgt = []

    for name, idx in zip(model_names, model_indices):

        if name in target_dict:
            src.append(idx)
            tgt.append(target_dict[name])

    return np.array(src), np.array(tgt)

def extract_points_from_indices(vertices, indices):
    return vertices[indices]

import numpy as np


def estimate_similarity_transform(source_points,
                                  target_points,
                                  allow_scale=True):
    """
    Compute similarity transform parameters (R, t, s)
    aligning source -> target.

    Returns
    -------
    R : (3,3)
    t : (3,)
    s : float
    """

    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)

    assert source.shape == target.shape, "Point sets must match"

    # centroids
    mu_src = source.mean(axis=0)
    mu_tgt = target.mean(axis=0)

    X = source - mu_src
    Y = target - mu_tgt

    # covariance
    H = X.T @ Y

    # SVD (Kabsch)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # reflection fix
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # scale
    if allow_scale:
        var_src = np.sum(X ** 2)
        s = np.sum(S) / var_src
    else:
        s = 1.0

    # translation
    t = mu_tgt - s * (R @ mu_src)

    return R, t, s


def apply_similarity_transform(points,
                                R,
                                t,
                                s=1.0):
    """
    Apply similarity transform:
        x' = s * R x + t

    Parameters
    ----------
    points : (N,3)
    R : (3,3)
    t : (3,)
    s : float

    Returns
    -------
    transformed : (N,3)
    """

    points = np.asarray(points, dtype=np.float64)

    return s * (points @ R.T) + t


def similarity_tf_dict(point_dict, R, t, s):
    aligned_point_dict = {}
    
    # Ensure t is (3,) then convert to column vector (3, 1)
    t_vec = np.array(t).flatten().reshape(3, 1) 
    
    for name, point in point_dict.items():
        # point is (3,), need (3, 1)
        pt = np.asarray(point).reshape(3, 1)
        
        # Apply transformation: P' = s*R*P + t
        # Ensure R is (3,3) and pt is (3,1)
        aligned_pt = (s * (R @ pt)) + t_vec
        aligned_point_dict[name] = aligned_pt.flatten()
        
    return aligned_point_dict