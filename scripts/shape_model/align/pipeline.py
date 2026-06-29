from dotenv import load_dotenv
import os
load_dotenv()

import json
import numpy as np

from src.geometry.vtk import load_vtp_mesh, save_vtp_mesh, get_vtp_vertices, create_polydata, add_point_field, create_polydata_w_lines
from src.geometry.geometry_np import test_LBO_spectrum, vertex_normals, compute_spectral_operators, save_spectral_data, load_spectral_data
from scripts.shape_model.align.src.fitting.icp_fitting import icp_fit
from scripts.shape_model.align.src.correspondence.correspondence import match_constrained
from scripts.shape_model.align.src.correspondence.vtk import save_matching_field
from scripts.shape_model.align.src.fitting.rigid import estimate_similarity_transform, apply_similarity_transform, similarity_tf_dict
from scripts.shape_model.align.src.model.model import StatisticalShapeModel, ShapeInstance
from scripts.shape_model.align.src.model.load import load_mat_ssm
from scripts.shape_model.align.src.data.load import load_landmarks
from scripts.shape_model.align.src.laplacian.laplacian import solve_laplacian_deformation

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
ALIGN = os.path.join(PROJECT_ROOT, "scripts", "shape_model", "align")
SHAPE_MODEL_MAT = os.path.join(PROJECT_ROOT, "scripts", "shape_model", "mat_model", "BabyFaceModel.mat")
MESH_VTP_PATH = os.path.join(PROJECT_ROOT, "US_samples", "US_samples", "clean", "clean.vtp")
LANDMARK_POSITIONS = os.path.join(PROJECT_ROOT, "US_samples", "US_samples", "clean", "landmark_positions.json")

surface = load_vtp_mesh(MESH_VTP_PATH)
surface_vertices = get_vtp_vertices(surface)
mean, eigenfunction, eigenvalues, model_landmarks= load_mat_ssm(SHAPE_MODEL_MAT)

shape_model = StatisticalShapeModel(
    mean=mean,
    eigenfunctions=eigenfunction,
    eigenvalues=eigenvalues,
    landmarks = model_landmarks
)
shape = ShapeInstance(shape_model)
coeffs = shape.fit(mean)
init_shape = shape.reconstruct()

print("mean.shape: ", mean.shape)
print("init_shape.shape: ", init_shape.shape)
print("eigenfunction.shape: ", eigenfunction.shape)
print("eigenvalues.shape: ", eigenvalues.shape)
print("Mean norm:", np.linalg.norm(mean))
print("Phi norm:", np.linalg.norm(eigenfunction))

vtp =create_polydata(mean.reshape(-1,3))
save_vtp_mesh(vtp, "mean.vtp")

vtp =create_polydata(init_shape.reshape(-1,3))
save_vtp_mesh(vtp, "init_shape.vtp")

vtp =create_polydata(surface_vertices)
save_vtp_mesh(vtp, "surface.vtp")

model_landmarks = shape.get_landmark_data()
target_landmarks, target_positions, target_names = load_landmarks(LANDMARK_POSITIONS)
print("model_landmarks: ", model_landmarks)
print("target_landmarks: ", target_landmarks)


print("--------Rigid Alignment test:")

surface_landmarks = [value for key, value in target_landmarks.items()]
curr_model_landmarks= [model_landmarks[key]["position"] for key, value in  target_landmarks.items()]



print("----------Step 1: (rigid alignment and model refinement)")
iter = 1
for i in range(iter):
    print("estimate")
    print("surface_landmarks: ", surface_landmarks)
    print("curr_model_landmarks: ", curr_model_landmarks)
    R,t,s = estimate_similarity_transform(surface_landmarks, curr_model_landmarks)

    print("apply")
    aligned_surface = apply_similarity_transform(surface_vertices, R, t, s )

    print("get coeffs")
    new_target_landmarks = similarity_tf_dict(target_landmarks, R, t, s)
    print("new target_points: ", new_target_landmarks)
    coeffs = shape.fit_landmarks(new_target_landmarks)
    refined_vertices = shape.reconstruct()
    print("refined_vertices.shape: ", refined_vertices.shape)

    curr_model_landmarks = shape.get_current_landmarks()

print("----------Save Results")
print("save aligned")
vtp = create_polydata(aligned_surface.reshape(-1,3))
save_vtp_mesh(vtp, "rigid_alignment.vtp")


print("save refined_vertices")
vtp = create_polydata(refined_vertices.reshape(-1,3))
save_vtp_mesh(vtp, "refined_vertices.vtp")


orig_langmarks = [value for key, value in target_landmarks.items()]
vtp = create_polydata(np.array(orig_langmarks).reshape(-1,3))
save_vtp_mesh(vtp, "orig_langmarks.vtp")

new_target_landmarks = [value for key, value in new_target_landmarks.items()]
vtp = create_polydata(np.array(new_target_landmarks).reshape(-1,3))
save_vtp_mesh(vtp, "new_target_landmarks.vtp")






print("----------Step 2: (Iterative Point registration under model constraints)")

print("refined_vertices.shape: ", refined_vertices.shape)
print("aligned_surface.shape: ", aligned_surface.shape)

src_normals = vertex_normals(refined_vertices.reshape(-1, 3))
tgt_normals = vertex_normals(aligned_surface.reshape(-1, 3))

print("src_normals.shape: ", src_normals.shape)
print("tgt_normals.shape: ", tgt_normals.shape)
#
valid_matches = match_constrained(refined_vertices.reshape(-1, 3), aligned_surface, src_normals, aligned_surface, 
                      dist_thresh = 0.4, normal_thresh_deg = 90)

print("valid_matches.shape", np.array(valid_matches).shape)

save_matching_field(refined_vertices.reshape(-1, 3), aligned_surface.reshape(-1, 3), valid_matches)
gammas = [0, 100, 10000, 20000, 30000, 50000, 80000, 100000]
nicp_vertices = {}
shapes = {}
from copy import deepcopy
for gamma in gammas:
    shape_fit = deepcopy(shape)
    coeffs = shape_fit.fit_to_matching(aligned_surface, valid_matches, learning_rate = 1, gamma=gamma, chi_clamp= None)
    nicp_refined_vertices = shape_fit.reconstruct()
    nicp_vertices[f"gamma_{gamma}"] = nicp_refined_vertices
    shapes[f"gamma_{gamma}"] = shape_fit
    vtp = create_polydata(nicp_refined_vertices.reshape(-1,3))
    save_vtp_mesh(vtp, f"nicp_refined_vertices_gamma_{gamma}.vtp")

shape_fit = deepcopy(shape)
coeffs = shape_fit.fit_to_matching(aligned_surface, valid_matches, learning_rate = 1, gamma=0, chi_clamp= None)
nicp_refined_vertices = shape_fit.reconstruct()
vtp = create_polydata(nicp_refined_vertices.reshape(-1,3))
save_vtp_mesh(vtp, "nicp_refined_vertices_free.vtp")

shape_fit = deepcopy(shape)
coeffs = shape_fit.fit_to_matching(aligned_surface, valid_matches, learning_rate = 1, gamma=0, chi_clamp= 0.999)
nicp_refined_vertices = shape_fit.reconstruct()
vtp = create_polydata(nicp_refined_vertices.reshape(-1,3))
save_vtp_mesh(vtp, "nicp_refined_vertices_clamped.vtp")





print("----------Step 3: (Regional Laplace Constrained Deformations)")
print("compute operators")
k_eig =20
""" 
L, M, spectral_evals, spectral_evecs = compute_spectral_operators_robust(
    nicp_vertices["gamma_80000"].reshape(-1, 3), k_eig=20, downsampling=False
)

save_spectral_data("spectral_data", L, M,  spectral_evals, spectral_evecs  )
"""
L, M, spectral_evals, spectral_evecs = load_spectral_data("spectral_data.npz")
print("got operators")

print("testing spectrum")      
test_LBO_spectrum(L, M, spectral_evals, spectral_evecs, k_eig )
print("L.shape: ", L.shape)
print("spectral_evecs.shape: ", spectral_evecs.shape)
print("spectral_evals.shape: ", spectral_evals.shape)


S_target = shapes["gamma_80000"].get_selection_matrix(lmk_name = "prn", radius = 0.3)
region_verts = shapes["gamma_80000"].region_vertices(lmk_name = "prn", radius = 0.3)
vtp = create_polydata(region_verts.reshape(-1,3))
save_vtp_mesh(vtp, "n_region_verts.vtp")

template_vertices = nicp_vertices["gamma_80000"].reshape(-1,3)
target_vertices = nicp_vertices["gamma_10000"].reshape(-1,3)

deformed_vertices = solve_laplacian_deformation(
    template_vertices, target_vertices, 
    S_target, L,
      lambda_reg=1.0)

vtp = create_polydata(deformed_vertices.reshape(-1,3))
save_vtp_mesh(vtp, "deformed_vertices.vtp")