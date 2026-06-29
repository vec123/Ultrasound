import os
import vtk
import numpy as np
from vtk.util import numpy_support
from scipy.spatial import KDTree

def create_radius_graph(input_vtp_path, output_vtp_path, mode='radial', k=5):
    """
    mode: 'radial' (Star graph to center) or 'knn' (K-Nearest Neighbors graph)
    """
    # 1. Read input
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_vtp_path)
    reader.Update()
    polydata = reader.GetOutput()
    
    num_orig_points = polydata.GetNumberOfPoints()
    points_arr = np.array([polydata.GetPoint(i) for i in range(num_orig_points)])
    
    # 2. Build Graph Structure
    new_points = vtk.vtkPoints()
    new_lines = vtk.vtkCellArray()
    
    if mode == 'radial':
        # Add center as point index 0
        center = np.mean(points_arr, axis=0)
        new_points.InsertNextPoint(center)
        for i in range(num_orig_points):
            new_points.InsertNextPoint(points_arr[i])
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, 0)
            line.GetPointIds().SetId(1, i + 1)
            new_lines.InsertNextCell(line)
            
    elif mode == 'knn':
        # Add all points as they are
        for i in range(num_orig_points):
            new_points.InsertNextPoint(points_arr[i])
        
        # Build kNN graph
        tree = KDTree(points_arr)
        _, indices = tree.query(points_arr, k=k+1) # k+1 because self is included
        
        for i in range(num_orig_points):
            for neighbor_idx in indices[i, 1:]:
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, i)
                line.GetPointIds().SetId(1, neighbor_idx)
                new_lines.InsertNextCell(line)

    # 3. Build final PolyData
    graph_polydata = vtk.vtkPolyData()
    graph_polydata.SetPoints(new_points)
    graph_polydata.SetLines(new_lines)
    
    # 4. Keep Landmark/Model Indices as scalar data
    # We add an array 'Original_Index' to visualize/track them in ParaView
    indices_arr = np.arange(num_orig_points)
    if mode == 'radial':
        indices_arr = np.insert(indices_arr, 0, -1) # -1 for center
        
    vtk_indices = numpy_support.numpy_to_vtk(indices_arr, deep=True, array_type=vtk.VTK_INT)
    vtk_indices.SetName("Original_Index")
    graph_polydata.GetPointData().AddArray(vtk_indices)
    
    # 5. Write result
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_vtp_path)
    writer.SetInputData(graph_polydata)
    writer.Write()
    print(f"[{mode.upper()}] Graph saved to: {output_vtp_path}")

# --- Usage ---
output_dir = "vtp_samples/radius_graphs"

# Radial (Star) Graph
create_radius_graph("vtp_samples/sample_01_bridge.vtp", 
                    os.path.join(output_dir, "sample_01_bridge_radial.vtp"), 
                    mode='radial')

# kNN Graph (k=5)
create_radius_graph("vtp_samples/sample_01_bridge.vtp", 
                    os.path.join(output_dir, "sample_01_bridge_knn.vtp"), 
                    mode='knn', k=5)