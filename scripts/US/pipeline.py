import os 
import glob
import torch

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")

INPUT_DIR = os.path.join(PROJECT_ROOT, "US_samples","US_samples","4","20_semanas","NRRD")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "US_samples", "US_samples","4","20_semanas", "VTK")
CACHE_DIR = os.path.join(PROJECT_ROOT, "US_samples", "US_samples","4","20_semanas", "Cache")
CONTOUR_VALUE = 70
K_EIG = 200
REDUCTION_FACTOR = 0.9
OPERATORS = False

from src.geometry.vtk import convert_single_nrrd, filter_largest_component, clean_mesh, reduce_mesh, create_polydata
from src.geometry.vtk import convert_single_nrrd, extract_arrays_from_polydata, add_point_field, save_vtp_mesh
from src.geometry.geometry import compute_operators, mesh_vertex_normals, get_operators
from src.geometry.geometry import to_basis, from_basis
from src.geometry.geometry import to_laplace_basis, from_laplace_basis
from src.geometry.geometry import get_operators
from src.geometry.geometry import normalize_field, compute_mean_curvature, compute_heat_kernel_signature

if __name__ == "__main__":

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
        
    # Find all .nrrd files in the target folder
    search_path = os.path.join(INPUT_DIR, "*.nrrd")
    nrrd_files = glob.glob(search_path)
    
    if not nrrd_files:
        raise ValueError(f"No .nrrd files found in {INPUT_DIR}")

    print(f"Found {len(nrrd_files)} .nrrd files to process.\n" + "-"*40)

    for nrrd_path in nrrd_files:
        base_name = os.path.splitext(os.path.basename(nrrd_path))[0]
        vtp_filename = f"{base_name}.vtp"

        print("nrrd to vtp")
        vtp = convert_single_nrrd(nrrd_path, CONTOUR_VALUE)

        print("clean mesh vtp")
        vtp = clean_mesh(vtp)

        print("keep largest component to vtp")
        vtp = filter_largest_component(vtp)
        
        print("reduce mesh vtp")
        vtp = reduce_mesh(vtp, reduction_factor=REDUCTION_FACTOR)

        mesh_verts, mesh_faces = extract_arrays_from_polydata(vtp)
        normals_np = mesh_vertex_normals(verts= mesh_verts, faces=mesh_faces)
        normals_np = normalize_field(normals_np)
       # mesh_verts = normalize_field(mesh_verts)

       # vtp = create_polydata(mesh_verts, mesh_faces)
        mean_curvature = compute_mean_curvature(torch.tensor(mesh_verts), torch.tensor(mesh_faces))

        if OPERATORS:
            cache_dir = CACHE_DIR # Define a persistent cache folder
            frames, mass, L, evals, evecs, gradX, gradY, features = get_operators(
                mesh_verts, 
                mesh_faces, 
                k_eig=K_EIG, 
                op_cache_dir=cache_dir
            )

        functions = {"vertices": mesh_verts,
                     "normals": normals_np}
        
        for key,function in functions.items():
            
            if OPERATORS:
                    coeffs = to_basis(function, evecs, mass)
                    func_reconstructed = from_basis(coeffs, evecs)
            print("function.shape[1]: ", function.shape)
            for i in  range(function.shape[1]):
                if isinstance(function, torch.Tensor):
                    function = function.detach().cpu().numpy()
                add_point_field(vtp, function[:,i], field_name=f"True {key} {i}")
                if OPERATORS:
                    add_point_field(vtp, func_reconstructed[:,i], field_name=f"recon {key} {i}")
        output = os.path.join(OUTPUT_DIR, vtp_filename)
        save_vtp_mesh(vtp, output)