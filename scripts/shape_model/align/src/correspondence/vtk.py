
import numpy as np
from src.geometry.vtk import save_vtp_mesh, create_polydata_w_lines

def save_matching_field(src_verts, tgt_verts, matches, filename="matching_field.vtp"):
    """
    matches: array of shape (N, 2) where col 0 is src_idx and col 1 is tgt_idx
    """
    src_np = src_verts
    tgt_np = tgt_verts
    
    matches = np.array(matches) 
    
    src_np = src_verts.cpu().numpy() if hasattr(src_verts, 'cpu') else src_verts
    tgt_np = tgt_verts.cpu().numpy() if hasattr(tgt_verts, 'cpu') else tgt_verts
    
     # Extract only the points that have valid matches
    src_matched = src_np[matches[:, 0].astype(int)]
    tgt_matched = tgt_np[matches[:, 1].astype(int)]
    
    # Create a line for each match
    # Paraview expects points and cells
    all_points = np.vstack([src_matched, tgt_matched])
    n_matches = len(matches)
    
    # Line indices: [0, n, 1, n+1, 2, n+2, ...]
    lines = np.column_stack([np.arange(n_matches), np.arange(n_matches) + n_matches])
    
    # Use your VTP creation function
    # Note: You may need a function that accepts 'lines' in addition to points
    vtp = create_polydata_w_lines(all_points, lines)
    save_vtp_mesh(vtp, filename)
