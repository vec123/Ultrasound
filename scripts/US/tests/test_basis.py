import os
import torch
import numpy as np
import potpourri3d as pp3d
from src.geometry.geometry import to_basis, from_basis
from src.geometry.geometry import to_laplace_basis, from_laplace_basis
from src.geometry.geometry import compute_operators
from src.geometry.vtk import create_polydata, add_point_field, save_vtk
from src.geometry.vtk import load_direct_vtp_mesh, extract_arrays_from_polydata

from src.models.GLO.geometric_basis import MSE_loss, GeometricBasis, fit_geometric_basis

from src.geometry.geometry import compute_operators, mesh_vertex_normals, get_operators, 

# --- Functions ---


if __name__ == "__main__":
    # --- Main Logic ---
    torch.set_default_dtype(torch.float64)
    load_dotenv()

    project_root = os.environ.get("PROJECT_ROOT")
    test_folder = os.path.join(project_root, "scripts", "US", "tests")

    vtp_shape_path = r"C:\Users\vic-b\Documents\Victors\Projects\Ultrasound\US_samples\US_samples\1\20_semanas\VTK\1-01_spectral.vtp"
    

    Laplacian_tf = True
    fitting = True
    k_eig = 200
    poly_data = load_direct_vtp_mesh(vtp_shape_path)
    mesh_verts, mesh_faces = extract_arrays_from_polydata(poly_data)
    normals_np = mesh_vertex_normals(verts= mesh_verts, faces=mesh_faces)

    func = torch.as_tensor(normals_np, dtype=torch.float64)
    func_mean = func.mean(dim=0, keepdim=True)
    func_std = func.std(dim=0, keepdim=True)
    func = (func - func_mean) / (func_std + 1e-8)
    
    #frames, mass, L, evals, evecs, gradX, gradY, features = compute_operators(mesh_verts, mesh_faces, k_eig)
    cache_dir = os.path.join(project_root, "cache", "ops") # Define a persistent cache folder
    frames, mass, L, evals, evecs, gradX, gradY, features = get_operators(
        mesh_verts, 
        mesh_faces, 
        k_eig=k_eig, 
        op_cache_dir=cache_dir
    )
    
    print("func.shape: ", func.shape)
    sigma_dim = 1

    # Initialize variables 
    alphas = None
    decay = 0.99
    decay_every = 100
    if Laplacian_tf:
        P = features['mean_curvature']
        if fitting:
            # alphas = torch.full((50, 1), 0.0, dtype=torch.float32)
            # alphas, sigmas = fit_geometric_basis_func(func, evecs, P, k_trunc=k_eig, learn_sigmas= True)  

            model = GeometricBasis(k_eig, dim = sigma_dim)
            model = fit_geometric_basis(func, model,  P, evecs, iterations=100, lr=0.1, 
                                        decay=decay, decay_every=decay_every )
            # Extract learned values
            basis = model.get_basis(P, evecs)
            sigmas = model.sigmas.detach()
            alphas = model.compute_coeffs(func, basis)
            print("alphas.shape: ", alphas.shape)
            print("sigmas.shape: ", sigmas.shape)
        else:
            sigma_val = 1.0# torch.full((50, 1), 0.1, dtype=torch.float32)
            alphas, basis = to_laplace_basis(func, evecs, mass, P, sigma_val)
            sigmas = torch.ones(k_eig, 1) * sigma_val
        
        with torch.no_grad():
            func_reconstructed = model(func, P, evecs)
   
    else:
        alphas = to_basis(func, evecs, mass)
        alphas_trunc = alphas.clone()
        func_reconstructed = from_basis(alphas, evecs)
        basis = evecs.unsqueeze(2)

    print("alphas.shape: ", alphas.shape)
    print("evecs.shape: ", evecs.shape)
    print("evals.shape: ", evals.shape)
    print("basis.shape: ", basis.shape)
    for i in range(0, func.shape[1]):
        loss = MSE_loss(func[:,i],func_reconstructed[:,i])
        print(f"{i} func - f_rec (components: {k_eig}): {loss}")


    # --- Visualization ---
    polydata = create_polydata(mesh_verts.numpy(), faces=mesh_faces.numpy())


    print("func.shape: ", func.shape)
    print("func_reconstructed.shape: ", func_reconstructed.shape)
    for i in range(0, func.shape[1]):
        add_point_field(polydata, normalize_field(func[:,i]), field_name=f"True {i}")
        add_point_field(polydata, normalize_field(func_reconstructed[:,i]), field_name=f"Reconstructed  {i}")


    save_vtk(polydata, os.path.join(test_folder, f"multi_field_analysis.vtk"))

    for i in range(k_eig):
        for k in range(0, sigma_dim):
          
            mode_i = basis[:,i, k]
            try:
                mode_i_np = mode_i.detach().cpu().numpy()
            except:
                mode_i = mode_i
            mode_norm = ((mode_i_np - mode_i_np.min()) / (mode_i_np.max() - mode_i_np.min() + 1e-8) * 100).astype(int)
            
            pd = create_polydata(mesh_verts.numpy(), faces=mesh_faces.numpy())
            add_point_field(pd, mode_norm, field_name=f"Mode_for_function_{k}")
            save_vtk(pd, os.path.join(test_folder, f"mode_{i:02d}.vtk"))