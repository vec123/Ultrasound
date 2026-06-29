import os
import numpy as np
import scipy.io as sio
import vtk
from vtk.util import numpy_support
from dotenv import load_dotenv
load_dotenv()

def get_rigid_transform(source_pts, target_pts):
    """Computes Scaling, Rotation, and Translation (Kabsch Algorithm)."""
    src_mean = np.mean(source_pts, axis=0)
    tgt_mean = np.mean(target_pts, axis=0)
    
    src_centered = source_pts - src_mean
    tgt_centered = target_pts - tgt_mean
    
    src_scale = np.sqrt(np.sum(src_centered**2) / len(source_pts))
    tgt_scale = np.sqrt(np.sum(tgt_centered**2) / len(target_pts))
    scale = tgt_scale / src_scale
    
    H = src_centered.T @ (tgt_centered / scale)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    T = tgt_mean - (scale * (R @ src_mean))
    return R, T, scale

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
BABYMODELPATH = os.path.join(PROJECT_ROOT, "scripts","shape_model", "BabyFaceModel.mat")
# --- 1. LOAD MODEL ---
mat_data = sio.loadmat(BABYMODELPATH, squeeze_me=True)
FaceModel = mat_data['BabyFaceModel']
refShape = FaceModel['refShape'].item()
meanDeformation = FaceModel['meanDeformation'].item()
triang = FaceModel['triang'].item()
lmks_vertsIND = FaceModel['landmark_verts'].item() - 1
landmark_names = FaceModel['landmark_names'].item()
trilist = (triang - 1).astype(np.int64, order='C')
mean_vertices = (meanDeformation + refShape.flatten(order='F')).reshape((3, -1), order='F').T

# --- 2. DEFINE TARGET LANDMARKS ---
# Example: Provide your target [x, y, z] for specific landmark names
print("landmark_names: ", landmark_names)
target_landmarks = {
    "prn":  [41.9958,32.4715,6.01142],
    "n": [43.7176,42.9437,8.73412],
    "enR": [49.657, 41.5853, 7.1040]
    # Add other landmarks here...
}

# --- 3. PREPARE POINTS FOR ALIGNMENT ---
source_indices = []
src_pts_list = []
tgt_pts_list = []

for i, name in enumerate(landmark_names):
    if name in target_landmarks:
        source_indices.append(lmks_vertsIND[i])
        src_pts_list.append(mean_vertices[lmks_vertsIND[i]])
        tgt_pts_list.append(target_landmarks[name])

R, T, s = get_rigid_transform(np.array(src_pts_list), np.array(tgt_pts_list))
tf_filename = "rigid_transform.npz"
np.savez(tf_filename, R=R, T=T, s=s)

print(f"Rigid transformation saved to {tf_filename}")
# --- 4. APPLY TRANSFORMATION ---
aligned_vertices = (s * (R @ mean_vertices.T)).T + T

def save_vtp(vertices, filename, connectivity):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(vertices.astype(np.float32), deep=True))
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    cells = vtk.vtkCellArray()
    cells.SetData(3, numpy_support.numpy_to_vtk(connectivity.flatten(), array_type=vtk.VTK_ID_TYPE))
    polydata.SetPolys(cells)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()

# Save both
save_vtp(mean_vertices, "original_mean_mesh.vtp", trilist)
save_vtp(aligned_vertices, "aligned_mean_mesh.vtp", trilist)

