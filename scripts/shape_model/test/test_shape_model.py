import os
import numpy as np
import scipy.io as sio
import scipy.stats as stats
import vtk
from vtk.util import numpy_support
from dotenv import load_dotenv
from src.geometry.vtk import create_polydata, add_point_field, save_vtp_mesh

load_dotenv()

# Helper function to mimic MATLAB's reshape(x, 3, [])
def rsp(x):
    return x.reshape((3, -1), order='F')

# --- LOAD BABY MODEL ---
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
SHAPE_MODEL_ROOT =os.path.join(PROJECT_ROOT, "scripts", "shape_model")
BABY_FACE_MODEL_PATH = os.path.join(SHAPE_MODEL_ROOT, "BabyFaceModel.mat")

mat_data = sio.loadmat(BABY_FACE_MODEL_PATH, squeeze_me=True)
FaceModel = mat_data['BabyFaceModel']

refShape = FaceModel['refShape'].item()
pctVar_per_eigen = FaceModel['pctVar_per_eigen'].item()
eigenValues = FaceModel['eigenValues'].item()
eigenFunctions = FaceModel['eigenFunctions'].item()
triang = FaceModel['triang'].item()
meanDeformation = FaceModel['meanDeformation'].item()

# Extract landmark data
lmks_vertsIND = FaceModel['landmark_verts'].item() - 1  # 0-indexed for Python
landmark_names = FaceModel['landmark_names'].item()

shapeMU = refShape.flatten(order='F')
trilist = (triang - 1).astype(np.int64, order='C')  

# %% GENERATE SYNTHETIC DATASET
nOfSamples = 15
var_gen = 99
nOfModes = np.where(np.cumsum(pctVar_per_eigen) > var_gen)[0][0] + 1

b = eigenValues[:nOfModes][:, np.newaxis] * (-3 + (3 + 3) * np.random.rand(nOfModes, nOfSamples)) * 1e6

output_dir = os.path.join(SHAPE_MODEL_ROOT,"vtp_samples")
os.makedirs(output_dir, exist_ok=True)

print(f"Saving modern VTK .vtp files with landmark heatmaps to '{output_dir}'...")

# Pre-convert raw triangle connections into a reusable VTK Data Array
vtk_connectivity = numpy_support.numpy_to_vtk(trilist.flatten(), deep=True, array_type=vtk.VTK_ID_TYPE)

for i in range(nOfSamples):
    #Shape reconstruction
    aux = meanDeformation + shapeMU + (eigenFunctions[:, :nOfModes] @ b[:, i])
    rec = rsp(aux)  
    vertices = rec.T.astype(np.float32, order='C')  # Shape: (N, 3)

    poly = create_polydata(vertices)
   
    
    # Calculate and Add Landmark Heatmaps (Point Data arrays)
    # We compute a Gaussian-like heatmap based on distance: exp(-dist^2 / (2 * sigma^2))
    # Adjust sigma to control how wide or localized the heatmap gradient looks
    sigma = 0.01 
    for lmk_idx, lmk_name in zip(lmks_vertsIND, landmark_names):
        # Clean up string type variations from MATLAB cell arrays
        if isinstance(lmk_name, np.ndarray) or not isinstance(lmk_name, str):
            lmk_name = str(lmk_name).strip()
        lmk_coords = vertices[lmk_idx] # Shape: (3,)
        # Compute Euclidean distance from all vertices to this landmark
        distances = np.linalg.norm(vertices - lmk_coords, axis=1)
        # Transform distances into an intensity heatmap (1 at landmark, tapering down to 0)
        heatmap_values = np.exp(-(distances ** 2) / (2 * (sigma ** 2))).astype(np.float32)
        poly = add_point_field(poly, heatmap_values, f"heatmap_{lmk_name}")

    save_vtp_mesh(poly, f"sample_{i+1:02d}.vtp")

print("Export complete! You can view the scalar fields under point attributes in ParaView.")