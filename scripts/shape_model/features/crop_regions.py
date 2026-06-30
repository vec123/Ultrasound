import os
import numpy as np
import scipy.io as sio
import vtk

from vtk.util import numpy_support
from dotenv import load_dotenv
from src.geometry.vtk import create_polydata, add_point_field, save_vtp_mesh
#from src.geometry.geometry_torch import compute_mean_curvature, compute_operators, vertex_normals
from src.geometry.geometry_np import(
    compute_grads,
    compute_mean_curvature, 
    compute_spectral_operators, 
    vertex_normals,
    test_LBO_spectrum
) 
# --- INITIALIZATION ---
load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
SHAPE_MODEL_ROOT =os.path.join(PROJECT_ROOT, "scripts", "shape_model")
BABY_FACE_MODEL_PATH = os.path.join(SHAPE_MODEL_ROOT, "mat_model", "BabyFaceModel.mat")

# --- LOAD DATA ---
mat_data = sio.loadmat(BABY_FACE_MODEL_PATH, squeeze_me=True)
FaceModel = mat_data['BabyFaceModel']

refShape = FaceModel['refShape'].item()
pctVar_per_eigen = FaceModel['pctVar_per_eigen'].item()
eigenValues = FaceModel['eigenValues'].item()
eigenFunctions = FaceModel['eigenFunctions'].item()
triang = FaceModel['triang'].item()
meanDeformation = FaceModel['meanDeformation'].item()
lmks_vertsIND = FaceModel['landmark_verts'].item() - 1
landmark_names = [str(n).strip() for n in FaceModel['landmark_names'].item()]

shapeMU = refShape.flatten(order='F')
trilist = (triang - 1).astype(np.int64, order='C')

# Pre-format connectivity for VTK: [3, p0, p1, p2, 3, p3, p4, p5, ...]
n_tris = trilist.shape[0]
cells_flat = np.hstack([np.full((n_tris, 1), 3), trilist]).flatten()

# --- PARAMETERS ---
nOfSamples = 200
var_gen = 99
sigma = 0.5
nOfModes = np.where(np.cumsum(pctVar_per_eigen) > var_gen)[0][0] + 1
region_config = {   'bridge': {'lmk_name': 'n', 'radius': 0.3},
                    'nose': {'lmk_name': 'prn', 'radius': 0.3},
                    'mouth': {'lmk_name': 'sl', 'radius': 0.3},
                    #'mouth_low': {'lmk_name': 'li', 'radius': 0.3},
                    #'mouth_top': {'lmk_name': 'ls', 'radius': 0.3},
                 }

# --- PROCESSING LOOP ---
print(f"Exporting to '{PROJECT_ROOT}'...")

for i in range(nOfSamples):
    # Setup directories
    shape_dir = os.path.join(SHAPE_MODEL_ROOT, "vtp_samples", f"shape_{i+1}")
    reg_dir = os.path.join(shape_dir, "regions")
    dataset_dir = os.path.join(SHAPE_MODEL_ROOT, "vtp_samples", "Dataset_faceparts_normalized")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(reg_dir, exist_ok=True)

    # 1. Generate random shape
    b = eigenValues[:nOfModes] * (-3 + 6 * np.random.rand(nOfModes)) * 1e6
    aux = meanDeformation + shapeMU + (eigenFunctions[:, :nOfModes] @ b)
    vertices = aux.reshape((3, -1), order='F').T.astype(np.float32)
    vertices -= vertices.mean(axis=0)
    max_val = np.abs(vertices).max()
    vertices /= max_val

    print("vertices: ", vertices)
    print("vertices.max(): ", vertices.max())
    print("vertices.min(): ", vertices.min())
    
    normals = vertex_normals(vertices, [])
    mean_curv = compute_mean_curvature(vertices)
    poly = create_polydata(vertices)

    """ 
    k_eig = 100
    L, M, evals, evecs = compute_spectral_operators(vertices, k_eig =k_eig)
    test_LBO_spectrum(L, M, evals, evecs, k_eig)

    gradX, gradY = compute_grads(vertices)          
  

    for i in range(gradX.shape[1]):
        poly = add_point_field(poly, gradX[:, i], f"gradX_eig_{i+1}")
    for i in range(gradY.shape[1]):
        poly = add_point_field(poly, gradX[:, i], f"gradY_eig_{i+1}")
    for i in range(evals.shape[0]):
        poly = add_point_field(poly, evecs[:, i], f"eigenvector_{i+1}")
    poly = add_point_field(poly, normals, "normals")
    poly = add_point_field(poly, mean_curv, "mean_curvature")  
    """
    #  Add Heatmaps
    for lmk_idx, lmk_name in zip(lmks_vertsIND, landmark_names):
        dists = np.linalg.norm(vertices - vertices[lmk_idx], axis=1)
        heatmap = np.exp(-(dists ** 2) / (2 * (sigma ** 2))).astype(np.float32)
        poly = add_point_field(poly, heatmap, f"heatmap_{lmk_name}")

    save_vtp_mesh(poly,  os.path.join(shape_dir, f"sample_{i+1:02d}.vtp"))
   
    # 5. Save Regions
    for reg_name, cfg in region_config.items():
        part_dir = os.path.join(dataset_dir, reg_name)
        os.makedirs(part_dir, exist_ok=True)

        lmk_idx = lmks_vertsIND[landmark_names.index(cfg['lmk_name'])]
        dists = np.linalg.norm(vertices - vertices[lmk_idx], axis=1)
        print(f"DEBUG: Shape {i+1} | Min dist: {dists.min():.4f} | Max dist: {dists.max():.4f} | Points within radius {cfg['radius']}: {np.sum(dists <= cfg['radius'])}")

        valid_indices = np.where(dists <= cfg['radius'])[0]
        id_map = {old: new for new, old in enumerate(valid_indices)}
        
        region_verts = vertices[valid_indices]
        region_verts -= region_verts.mean(axis=0)
        max_val = np.abs(region_verts).max()
        region_verts /= max_val
        poly = create_polydata(region_verts)

        save_vtp_mesh(poly, os.path.join(part_dir, f"shape_{i}.vtp"))
        #save_vtp_mesh(poly, os.path.join(dataset_dir, f"{reg_name}.vtp"))
        
print("Export complete.")