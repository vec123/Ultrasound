from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
import glob
import os


from scripts.shape_model.features.compute_tensors import compute_vertex_normals, compute_mean_curvature, compute_distance_to_centroid, compute_gaussian_curvature
from scripts.shape_model.features.compute_tensors import compute_shape_index, compute_curvedness, compute_principal_curvature_directions,compute_local_density
from scripts.shape_model.features.compute_tensors import  compute_rel_distance_vectors, project_relative_positions
from src.geometry.vtk import load_vtp_mesh, get_vtp_vertices, add_point_field, save_vtp_mesh
from dotenv import load_dotenv
load_dotenv()

master_key = jax.random.PRNGKey(42)

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
VTP_PATH = os.path.join(PROJECT_ROOT, "Dataset_faces", "shape_1", "sample_01.vtp")

vtp = load_vtp_mesh(VTP_PATH)
verts = get_vtp_vertices(vtp)
print("verts.shape: ", verts.shape)

normals = compute_vertex_normals(verts, k=100)
print("normals.shape: ",normals.shape)
for i in range(normals.shape[1]):
    vtp = add_point_field(vtp, normals[:,i], f"Normals_{i}")

""" 
in_comp, orth_comp = project_relative_positions(verts, normals)
print("in_comp.shape: ",in_comp.shape)
print("orth_comp.shape: ",orth_comp.shape)
for i in range(in_comp.shape[1]):
    vtp = add_point_field(vtp, in_comp[:,i], f"in_comp_{i}")
    vtp = add_point_field(vtp, orth_comp[:,i], f"orth_comp_{i}")
"""

mean_curvature = compute_mean_curvature(verts, k=100)
print("mean_curvature.shape: ",mean_curvature.shape)
vtp = add_point_field(vtp, mean_curvature, "mean_curvature")

dist_to_centroid = compute_distance_to_centroid(verts, k=100)
print("dist_to_centroid.shape: ",dist_to_centroid.shape)
vtp = add_point_field(vtp, dist_to_centroid, "dist_to_centroid")

gaussian_curvature = compute_gaussian_curvature(verts, k=100)
print("gaussian_curvature.shape: ",gaussian_curvature.shape)
vtp = add_point_field(vtp, gaussian_curvature, "gaussian_curvature")

k1, k2, _ ,_ = compute_principal_curvature_directions(verts)

shape_idx = compute_shape_index(k1, k2)
print("shape_idx.shape: ",shape_idx.shape)
vtp = add_point_field(vtp, shape_idx, "shape_idx")

curvedness = compute_curvedness(k1, k2)
print("curvedness.shape: ",curvedness.shape)
vtp = add_point_field(vtp, curvedness, "curvedness")

local_density = compute_local_density(verts)
print("local_density.shape: ",local_density.shape)
vtp = add_point_field(vtp, local_density, "local_density")

rel_dist_vectors = compute_rel_distance_vectors(verts)
print("rel_dist_vectors.shape: ",rel_dist_vectors.shape)
out_path = os.path.join(PROJECT_ROOT, "Dataset_faces", "shape_1", "test_tensor_features.vtp")
save_vtp_mesh(vtp,out_path)