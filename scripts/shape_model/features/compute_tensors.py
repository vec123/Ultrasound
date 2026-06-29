import numpy as np
from scipy.spatial import KDTree
import potpourri3d as pp3d


def cross(vec_A, vec_B):
    return np.cross(vec_A, vec_B, dim=-1)


def dot(vec_A, vec_B):
    return np.sum(vec_A * vec_B, dim=-1)


def get_knn(points, k=10):
    tree = KDTree(points)
    return tree.query(points, k=k+1)[1][:, 1:] # Exclude the point itself

def project_to_tangent(vecs, unit_normals):
    dots = dot(vecs, unit_normals)
    return vecs - unit_normals * dots.unsqueeze(-1)

def neighborhood_normal(points):
    # points: (N, K, 3) array of neighborhood psoitions
    # points should be centered at origin
    # out: (N,3) array of normals
    # numpy in, numpy out
    (u, s, vh) = np.linalg.svd(points, full_matrices=False)
    normal = vh[:,2,:]
    return normal / np.linalg.norm(normal,axis=-1, keepdims=True)

def mesh_vertex_normals(verts, faces):
    # numpy in / out
    #face_n = toNP(face_normals(torch.tensor(verts), torch.tensor(faces))) # ugly torch <---> numpy
    face_n = faces
    vertex_normals = np.zeros(verts.shape)
    for i in range(3):
        np.add.at(vertex_normals, faces[:,i], face_n)

    vertex_normals = vertex_normals / (np.linalg.norm(vertex_normals,axis=-1,keepdims=True) +1e-8)

    return vertex_normals

# --- SCALAR FEATURES (l=0) ---

def compute_mean_curvature(verts, faces=None, k=30):
    """
    Unified function for mean curvature estimation.
    verts: (N, 3) tensor
    faces: (M, 3) tensor or None
    k: neighborhood size for point cloud mode
    """
    verts_np = verts
    
    # CASE 1: Mesh Input (Using Cotan Laplacian)
    if faces is not None:
        faces_np = faces
        L = pp3d.cotan_laplacian(verts_np, faces_np)
        lap_pos = L @ verts_np
        H = np.linalg.norm(lap_pos, axis=1) / 2.0
        
    # CASE 2: Point Cloud Input (Using PCA on local neighbors)
    else:
        tree = KDTree(verts_np)
        # Query k+1 and exclude the point itself
        knn_idx = tree.query(verts_np, k=k+1)[1][:, 1:]
        knn_pts = verts_np[knn_idx]
        
        # Center points relative to their own neighborhoods
        centroids = np.mean(knn_pts, axis=1, keepdims=True)
        centered = knn_pts - centroids
        
        # Compute Covariance matrices (N, 3, 3)
        cov = np.einsum('nki,nkj->nij', centered, centered) / k
        
        # PCA: Smallest eigenvalue of the covariance matrix
        eigenvalues = np.linalg.eigvalsh(cov)
        lambda_0 = eigenvalues[:, 0]
        sum_eigen = np.sum(eigenvalues, axis=1)
        
        # Normalize curvature estimate
        H = lambda_0 / (sum_eigen + 1e-8)
        def normalize_robust(H, p_min=2, p_max=98):
            h_min = np.percentile(H, p_min)
            h_max = np.percentile(H, p_max)
            # Clip values outside the range
            H_clipped = np.clip(H, h_min, h_max)
            return (H_clipped - h_min) / (h_max - h_min + 1e-8)
        H = normalize_robust(H)
    return H

def compute_vertex_normals(verts, faces=None, k=30):
    """
    Unified function to compute normals for both meshes and point clouds.
    If faces are provided, uses area-weighted face normals.
    If no faces are provided, uses Gaussian-weighted PCA.
    """
    verts = np.asarray(verts)
    
    # CASE 1: Mesh Input
    if faces is not None:
        faces = np.asarray(faces)
        normals = np.zeros_like(verts, dtype=np.float64)
        
        # Calculate face normals and areas
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        
        # Face normal = (v1-v0) cross (v2-v0)
        face_normals = np.cross(v1 - v0, v2 - v0)
        
        # Accumulate face normals to vertices (Area-weighted)
        for i in range(3):
            np.add.at(normals, faces[:, i], face_normals)
            
    # CASE 2: Point Cloud Input (Weighted PCA)
    else:
        tree = KDTree(verts)
        dist, idx = tree.query(verts, k=k+1)
        neigh_pts = verts[idx[:, 1:]]
        centered = neigh_pts - verts[:, np.newaxis, :]
        
        # Gaussian weights for stability
        sigma = np.mean(dist[:, 1:]) * 0.5 
        weights = np.exp(-np.sum(centered**2, axis=-1) / (2 * sigma**2 + 1e-8))
        
        # Weighted Covariance
        weighted_centered = centered * weights[..., np.newaxis]
        cov = np.einsum('nki,nkj->nij', weighted_centered, centered)
        
        # SVD for normal
        _, _, vh = np.linalg.svd(cov)
        normals = vh[:, 2, :]
        
        # Orient towards centroid
        centroid = np.mean(verts, axis=0)
        to_centroid = verts - centroid
        flip = np.sum(normals * to_centroid, axis=-1) < 0
        normals[flip] *= -1

    # Final normalization
    norm_mag = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / (norm_mag + 1e-8)

def compute_distance_to_centroid(points, k=10, weighted=True, normalize=True):
    """
    Computes the distance from each point to the centroid of its k-nearest neighbors.
    
    Args:
        points: (N, 3) array
        k: Neighborhood size
        weighted: If True, uses inverse-distance weighting
        normalize: If True, scales output to [0, 1] range using Min-Max normalization
    
    Returns:
        (N,) array of distances (normalized if requested)
    """
    pts = np.asarray(points)
    tree = KDTree(pts)
    
    # Query k neighbors (excluding the point itself)
    dists, indices = tree.query(pts, k=k+1)
    dists = dists[:, 1:]
    neighbors = pts[indices[:, 1:]]
    
    if weighted:
        # Inverse distance weighting
        weights = 1.0 / (dists + 1e-8)
        weights /= np.sum(weights, axis=1, keepdims=True)
        centroids = np.sum(neighbors * weights[..., np.newaxis], axis=1)
    else:
        centroids = np.mean(neighbors, axis=1)
    
    raw_distances = np.linalg.norm(pts - centroids, axis=1)
    
    if normalize:
        min_dist = np.min(raw_distances)
        max_dist = np.max(raw_distances)
        
        # Avoid division by zero if point cloud is perfectly flat
        denom = max_dist - min_dist
        if denom < 1e-8:
            return np.zeros_like(raw_distances)
            
        return (raw_distances - min_dist) / denom
    
    return raw_distances

def compute_local_density(points, k=10):
    """
    Computes local density as the inverse of the average distance to k-nearest neighbors.
    High value = high density (points are closer together).
    """
    pts = np.asarray(points)
    tree = KDTree(pts)
    dist, _ = tree.query(pts, k=k+1)
    
    # Mean distance to k neighbors (excluding the point itself at index 0)
    avg_dist = np.mean(dist[:, 1:], axis=1)
    
    # Inverse distance for density (add epsilon to avoid division by zero)
    return 1.0 / (avg_dist + 1e-6)


def compute_gaussian_curvature(points, k=15):
    """
    Estimates Gaussian Curvature via local quadric surface fitting.
    points: (N, 3) array
    k: neighborhood size
    """
    pts = np.asarray(points)
    n = pts.shape[0]
    
    # 1. Get neighbors
    tree = KDTree(pts)
    indices = tree.query(pts, k=k+1)[1][:, 1:]
    neighbors = pts[indices] # (N, k, 3)
    
    # 2. Local coordinate system: Center neighbors
    pts_centered = neighbors - pts[:, np.newaxis, :]
    
    # 3. Fit a quadric surface: z = ax^2 + bxy + cy^2
    # This requires local PCA to align the tangent plane with the xy-plane
    gaussian_curvatures = np.zeros(n)
    
    for i in range(n):
        # Local Basis via SVD
        _, _, vh = np.linalg.svd(pts_centered[i])
        normal = vh[2, :]
        basis = vh[:2, :] # Tangent plane basis
        
        # Project neighbors to local tangent plane
        p_local = pts_centered[i] @ basis.T # (k, 2)
        z = pts_centered[i] @ normal        # (k, 1)
        
        # Solve Ax = b for coefficients [a, b, c]
        # z = ax^2 + bxy + cy^2
        x, y = p_local[:, 0], p_local[:, 1]
        A = np.stack([x**2, x*y, y**2], axis=1)
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        a, b, c = coeffs
        
        # Gaussian Curvature K = (4ac - b^2) / (1 + z_x^2 + z_y^2)^2
        # At the origin (the point itself), z_x and z_y are 0
        gaussian_curvatures[i] = (4*a*c - b**2)
        
        def normalize_robust(H, p_min=2, p_max=98):
            h_min = np.percentile(H, p_min)
            h_max = np.percentile(H, p_max)
            # Clip values outside the range
            H_clipped = np.clip(H, h_min, h_max)
            return (H_clipped - h_min) / (h_max - h_min + 1e-8)
        gaussian_curvatures = normalize_robust(gaussian_curvatures)
    return gaussian_curvatures

# --- VECTOR FEATURES (l=1) ---

def compute_rel_distance_vectors(points, k=10):
    indices = get_knn(points, k)
    # Returns (N, k, 3)
    return points[:, np.newaxis, :] - points[indices]


def compute_principal_curvature_directions(points, k=20):
    pts = np.asarray(points)
    n = pts.shape[0]
    k1, k2 = np.zeros(n), np.zeros(n)
    v1, v2 = np.zeros((n, 3)), np.zeros((n, 3))
    
    tree = KDTree(pts)
    _, indices = tree.query(pts, k=k+1)
    
    for i in range(n):
        neighbors = pts[indices[i, 1:]] - pts[i]
        
        # 1. Local Tangent Basis (SVD)
        # Normal is the eigenvector corresponding to the smallest eigenvalue
        u, s, vh = np.linalg.svd(neighbors)
        normal = vh[2, :]
        basis = vh[:2, :] # Tangent directions [t1, t2]
        
        # 2. Local coordinates (u, v, w)
        coords = neighbors @ vh.T 
        u_vals, v_vals, w_vals = coords[:, 0], coords[:, 1], coords[:, 2]
        
        # 3. Fit Full Quadric: w = c1*u^2 + c2*uv + c3*v^2
        # (Assuming centered at origin, so linear terms are 0)
        A = np.stack([0.5 * u_vals**2, u_vals * v_vals, 0.5 * v_vals**2], axis=1)
        coeffs, _, _, _ = np.linalg.lstsq(A, w_vals, rcond=None)
        c1, c2, c3 = coeffs
        
        # 4. Weingarten Map (Shape Operator) S = [[c1, c2], [c2, c3]]
        S = np.array([[c1, c2], [c2, c3]])
        
        # 5. Eigen-decomposition of S
        eigvals, eigvecs = np.linalg.eigh(S)
        
        # Sorted: k1 >= k2
        k1[i], k2[i] = eigvals[1], eigvals[0]
        v1[i] = eigvecs[:, 1] @ basis
        v2[i] = eigvecs[:, 0] @ basis
        
    return k1, k2, v1, v2

def compute_shape_index(k1, k2):
    """
    Computes Shape Index (range: -1 to 1).
    Formula: (2/pi) * atan((k2 + k1) / (k2 - k1))
    """
    # Use np.arctan2 for numerical stability
    shape_index = (2.0 / np.pi) * np.arctan2(k2 + k1, k2 - k1 + 1e-10)
    return shape_index

def compute_curvedness(k1, k2):
    """
    Computes Curvedness (magnitude of bending).
    Formula: sqrt((k1^2 + k2^2) / 2)
    """
    return np.sqrt((k1**2 + k2**2) / 2.0)

def compute_curvature_gradient(points, k=10):
    """
    Computes the gradient of Mean Curvature across the point cloud.
    points: (N, 3) array
    k: neighborhood size
    
    Returns:
    grad_H: (N, 3) array representing the gradient vector of H
    """
    pts = np.asarray(points)
    n = pts.shape[0]
    
    # 1. Compute Mean Curvature as the scalar field
    # (Reusing your existing logic)
    H = compute_mean_curvature(pts, faces=None, k=k)
    
    # 2. Get Neighbors
    tree = KDTree(pts)
    indices = tree.query(pts, k=k+1)[1][:, 1:]
    
    grad_H = np.zeros((n, 3))
    
    for i in range(n):
        # Local neighbors and curvature difference
        neigh_pts = pts[indices[i]]
        dH = H[indices[i]] - H[i] # Scalar curvature differences
        
        # Local tangent basis via PCA
        diffs = neigh_pts - pts[i]
        _, _, vh = np.linalg.svd(diffs)
        basis = vh[:2, :] # Tangent basis (2, 3)
        
        # Project neighbor displacements into tangent plane
        # (k, 3) @ (3, 2) -> (k, 2)
        proj_pts = diffs @ basis.T
        
        # Solve for gradient (g_u, g_v) in tangent space
        # dH = g_u * u + g_v * v
        # Linear Least Squares: proj_pts @ [g_u, g_v]^T = dH
        grad_uv, _, _, _ = np.linalg.lstsq(proj_pts, dH, rcond=None)
        
        # Project back to 3D: grad = g_u * u_basis + g_v * v_basis
        grad_H[i] = grad_uv @ basis
        
    return grad_H


def compute_local_normal_dispersion(points, normals, k=10):
    """
    Computes local normal dispersion.
    points: (N, 3) array
    normals: (N, 3) array (pre-computed unit vectors)
    k: neighborhood size
    
    Returns:
    dispersion: (N,) array where 0 = perfect alignment, 1 = maximum dispersion
    """
    pts = np.asarray(points)
    nrms = np.asarray(normals)
    
    # 1. Get neighbors
    tree = KDTree(pts)
    indices = tree.query(pts, k=k+1)[1][:, 1:]
    
    # 2. Extract neighbor normals
    neigh_normals = nrms[indices] # (N, k, 3)
    
    # 3. Compute mean normal in the neighborhood
    # Note: We take absolute value if normals are unoriented to avoid 
    # cancellation of antiparallel vectors
    mean_normal = np.mean(neigh_normals, axis=1)
    
    # 4. Dispersion is related to the magnitude of the mean vector
    # If normals are perfectly aligned, norm(mean) -> 1
    # If normals are perfectly dispersed, norm(mean) -> 0
    mean_norm_mag = np.linalg.norm(mean_normal, axis=1)
    
    # Dispersion (0 to 1 scale)
    dispersion = 1.0 - mean_norm_mag
    
    return np.clip(dispersion, 0, 1)


def project_relative_positions(points, normals, k=10):
    """
    Decomposes neighbor displacements into tangent and normal components.
    
    points: (N, 3) array
    normals: (N, 3) array (should be unit length)
    k: neighborhood size
    
    Returns:
    tangent_comp: (N, k, 3) relative vectors in the tangent plane
    normal_comp: (N, k, 3) relative vectors along the normal
    """
    pts = np.asarray(points)
    nrms = np.asarray(normals)
    n = pts.shape[0]
    
    # 1. Get neighbors
    tree = KDTree(pts)
    indices = tree.query(pts, k=k+1)[1][:, 1:]
    
    # 2. Compute relative vectors v_ij = p_j - p_i
    # neighbors shape: (N, k, 3)
    rel_vecs = pts[indices] - pts[:, np.newaxis, :]
    
    # 3. Project onto normal (Normal component)
    # nrm_comp = (v_ij dot n_i) * n_i
    # We use einsum for efficient batch dot products
    dots = np.einsum('nki,ni->nk', rel_vecs, nrms)
    normal_comp = dots[:, :, np.newaxis] * nrms[:, np.newaxis, :]
    
    # 4. Project onto tangent plane (Tangent component)
    # tang_comp = v_ij - nrm_comp
    tangent_comp = rel_vecs - normal_comp
    
    return tangent_comp, normal_comp

def sort_neighborhood(rel_vecs, method='distance'):
    """
    Sorts relative distance vectors (N, K, 3) to create canonical order.
    
    rel_vecs: (N, K, 3) array of displacement vectors
    method: 'distance' (Euclidean), 'projected' (Along normal), or 'signed_dist'
    
    Returns:
    sorted_rel_vecs: (N, K, 3) sorted displacement vectors
    """
    if method == 'distance':
        # Sort by Euclidean norm (closest to furthest)
        dists = np.linalg.norm(rel_vecs, axis=-1)
        sort_idx = np.argsort(dists, axis=1)
        
    elif method == 'projected':
        # Sort by the magnitude of the displacement along the normal 
        # (requires 'normal' input if you want absolute projection)
        # Assuming we sort by the raw component magnitude
        dists = np.abs(np.linalg.norm(rel_vecs, axis=-1)) 
        sort_idx = np.argsort(dists, axis=1)
        
    elif method == 'angle':
        # Sort by angular position (e.g., around the normal vector)
        # Useful for capturing rotational symmetry
        angles = np.arctan2(rel_vecs[..., 1], rel_vecs[..., 0])
        sort_idx = np.argsort(angles, axis=1)
    
    else:
        raise ValueError("Unknown sorting method")

    # Apply the sort indices using advanced indexing
    # We use np.take_along_axis for cleaner syntax on multi-dim arrays
    sorted_rel_vecs = np.take_along_axis(rel_vecs, sort_idx[..., np.newaxis], axis=1)
    
    return sorted_rel_vecs