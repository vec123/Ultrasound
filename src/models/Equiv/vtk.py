import vtk
from vtk.util import numpy_support
import numpy as np
import jraph
import jax.numpy as jnp
import os


def save_single_graph_as_vtp(graph: jraph.GraphsTuple, filename: str):
    """
    Saves a single jraph.GraphsTuple as a .vtp file.
    
    Args:
        graph: A jraph.GraphsTuple representing a single graph.
        filename: Full path where the .vtp file will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 1. Create VTK points
    # Convert nodes to numpy and then to VTK points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(np.array(graph.nodes)))
        
    # 2. Create Cell Connectivity
    # VTK expects cells in the format [n_points, p0, p1, ..., n_points, p0, p1, ...]
    num_edges = int(graph.n_edge[0])
    cells_array = np.empty((num_edges, 3), dtype=np.int64)
    cells_array[:, 0] = 2  # Each line has 2 points
    cells_array[:, 1] = np.array(graph.senders)
    cells_array[:, 2] = np.array(graph.receivers)
    
    # Create VTK CellArray from the flat numpy array
    cells = vtk.vtkCellArray()
    cells.SetCells(num_edges, numpy_support.numpy_to_vtkIdTypeArray(cells_array.ravel()))
        
    # 3. Create PolyData
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(cells)
    
    # 4. Write
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()
    
    print(f"Successfully saved graph to {filename}")

    
def save_graphs_as_vtp(graph: jraph.GraphsTuple, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # Pre-calculate edge starts to avoid re-summing inside the loop
    # n_edge tracks how many edges each graph has
    edge_counts = graph.n_edge
    edge_offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(edge_counts[:-1])])
    node_offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(graph.n_node[:-1])])
    
    for i in range(len(graph.n_node)):
        # Extract subset of edges and nodes
        start_e, end_e = edge_offsets[i], edge_offsets[i] + edge_counts[i]
        start_n, end_n = node_offsets[i], node_offsets[i] + graph.n_node[i]
        
        subset_senders = graph.senders[start_e:end_e] - start_n
        subset_receivers = graph.receivers[start_e:end_e] - start_n
        
        # 1. Create VTK points
        # numpy_support.numpy_to_vtk is much faster than manual loop
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_support.numpy_to_vtk(np.array(graph.nodes[start_n:end_n])))
            
        # 2. Create Cell Connectivity
        # VTK needs an array: [2, s0, r0, 2, s1, r1, ...]
        num_edges = int(edge_counts[i])
        cells_array = np.empty((num_edges, 3), dtype=np.int64)
        cells_array[:, 0] = 2  # Each line has 2 points
        cells_array[:, 1] = subset_senders
        cells_array[:, 2] = subset_receivers
        
        # Create VTK CellArray from flat numpy array
        cells = vtk.vtkCellArray()
        cells.SetCells(num_edges, numpy_support.numpy_to_vtkIdTypeArray(cells_array.ravel()))
            
        # 3. Create PolyData
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetLines(cells)
        
        # 4. Write
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(os.path.join(output_dir, f"graph_{i}.vtp"))
        writer.SetInputData(polydata)
        writer.Write()
        
    print(f"Successfully saved {len(graph.n_node)} graphs to {output_dir}")