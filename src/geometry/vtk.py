import os
import vtk
import numpy as np
from vtk.util import numpy_support

# --- Integration Helpers ---
def load_vtp_mesh(vtp_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    return reader.GetOutput()

def get_vtp_vertices(polydata):

    points = polydata.GetPoints()
    
    # Convert vtkPoints to a numpy array
    # vtk_array_to_numpy provides a direct view or copy of the underlying data
    vertices = numpy_support.vtk_to_numpy(points.GetData())
    
    return vertices

def load_vtu(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
        
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()  # This executes the reader
    
    # Return the actual data object, not the port
    return reader.GetOutput()
def convert_vtu_to_vtp_vtk(vtu_data):
    """
    Converts a VTU object to VTP, with internal checks to ensure 
    valid input and non-empty output.
    """
    # 1. Check if the input is valid
    if vtu_data is None:
        raise ValueError("Input vtu_data is None.")
        
    print(f"Input object type: {vtu_data.GetClassName()}")
    print(f"Input points: {vtu_data.GetNumberOfPoints()}")
    
    if vtu_data.GetNumberOfPoints() == 0:
        raise ValueError("Input VTU has 0 points. Conversion aborted.")

    # 2. Setup the filter
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(vtu_data)
    surface_filter.Update()
    
    # 3. Retrieve the output
    vtp_output = surface_filter.GetOutput()
    
    # 4. Check if the filter actually produced something
    if vtp_output is None or vtp_output.GetNumberOfPoints() == 0:
        raise RuntimeError("Surface extraction failed: The resulting VTP is empty.")
    
    print(f"Conversion successful. Output points: {vtp_output.GetNumberOfPoints()}")
    
    return vtp_output


def extract_arrays_from_polydata(poly_data):
    points = numpy_support.vtk_to_numpy(poly_data.GetPoints().GetData())
    cells = numpy_support.vtk_to_numpy(poly_data.GetPolys().GetConnectivityArray())
    return points, cells.reshape(-1, 3)


def filter_largest_component(poly_data):
    """Loads a VTP, extracts only the largest connected component, and saves it."""

    print(f"Filter largest component for mesh with {poly_data.GetNumberOfPoints()} points.")

    # 2. Extract the largest connected component
    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(poly_data)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()
    
    largest_mesh = connectivity.GetOutput()
    
    print(f"largest component has {largest_mesh.GetNumberOfPoints()} points.")

    return largest_mesh

def clean_mesh(polydata):
    """
    Isolates the single largest connected component, removes topological defects,
    and decimates the triangle density by the specified reduction factor.
    """
    
    # 1. Initialize VTP Reader
    raw_mesh = polydata
    
    orig_verts = raw_mesh.GetNumberOfPoints()
    orig_cells = raw_mesh.GetNumberOfCells()
    print(f"Original Count : {orig_verts:,} vertices | {orig_cells:,} triangles")

    # 2. Isolate the absolute largest component (strips away tiny floating noise)
    print("Applying Connectiviy filter...")
    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(raw_mesh)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()
    
    # 3. Clean up topology (merges coincident points, eliminates zero-area triangles)
    print("Cleaning mesh topology and removing degenerate geometry...")
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(connectivity.GetOutput())
    cleaner.Update()
    out = cleaner.GetOutput()
    return out

def reduce_mesh(polydata, reduction_factor=0.90):
    # 4. Decimate the mesh using Quadric Error Metrics
    # Keeps structural landmarks (edges, curves) while aggressively thinning flat zones.

    raw_mesh = polydata
    orig_verts = raw_mesh.GetNumberOfPoints()
    orig_cells = raw_mesh.GetNumberOfCells()
    print(f"original Count: {orig_verts:,} vertices | {orig_cells:,} triangles")

    print(f"Decimating triangles by {reduction_factor * 100:.1f}%...")
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(raw_mesh)
    decimate.SetTargetReduction(reduction_factor)
    decimate.Update()
    
    reduced_mesh = decimate.GetOutput()
    
    final_verts = reduced_mesh.GetNumberOfPoints()
    final_cells = reduced_mesh.GetNumberOfCells()
    
    print("-" * 60)
    print(f"Optimized Count: {final_verts:,} vertices | {final_cells:,} triangles")
    print(f"Data Reduction : {((orig_verts - final_verts) / orig_verts) * 100:.2f}% vertices removed.")
    print("-" * 60)

    return reduced_mesh

def create_polydata(points, faces=None):
    """Creates a VTK PolyData object from points and optional faces."""
    polydata = vtk.vtkPolyData()
    
    # Set Points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(points.astype(np.float32)))
    polydata.SetPoints(vtk_points)
    
    # Set Faces
    if faces is not None:
        # Create a vtkIdTypeArray for cell connectivity
        # Format: [3, i1, i2, i3, 3, j1, j2, j3, ...]
        n_faces = faces.shape[0]
        # Prepend '3' (number of vertices per triangle) to every row
        cells_flat = np.hstack([np.ones((n_faces, 1), dtype=np.int64) * 3, faces.astype(np.int64)])
        
        # Convert to vtkIdTypeArray
        cell_array = vtk.vtkCellArray()
        # Newer VTK versions prefer SetCells with an offset array, 
        # but this conversion is the standard way to feed legacy-style data:
        cell_array.SetCells(n_faces, numpy_support.numpy_to_vtk(cells_flat.flatten(), deep=True, array_type=vtk.VTK_ID_TYPE))
        
        polydata.SetPolys(cell_array)
    else:
        # Fallback to points/vertices
        vertices = vtk.vtkCellArray()
        for i in range(len(points)):
            vertices.InsertNextCell(1, [i])
        polydata.SetVerts(vertices)
        
    return polydata

def create_polydata_w_lines(points, lines):
    """
    points: (N, 3) array
    lines: (M, 2) array of index pairs [start_idx, end_idx]
    """
    polydata = vtk.vtkPolyData()
    
    # 1. Set Points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(points.astype(np.float32)))
    polydata.SetPoints(vtk_points)
    
    # 2. Set Lines
    # VTK expects lines in the format [2, idx1, idx2, 2, idx3, idx4, ...]
    n_lines = lines.shape[0]
    lines_flat = np.hstack([np.ones((n_lines, 1), dtype=np.int64) * 2, lines.astype(np.int64)])
    
    cell_array = vtk.vtkCellArray()
    cell_array.SetCells(n_lines, numpy_support.numpy_to_vtk(lines_flat.flatten(), deep=True, array_type=vtk.VTK_ID_TYPE))
    
    polydata.SetLines(cell_array)
    return polydata

def add_point_field(polydata, field_data, field_name="field"):
    """Adds a scalar or vector field to the existing PolyData."""
    vtk_array = numpy_support.numpy_to_vtk(field_data, deep=True)
    vtk_array.SetName(field_name)
    polydata.GetPointData().AddArray(vtk_array)
    return polydata

def save_vtp_mesh(polydata, vtk_path, binary=True):
    """Saves the PolyData object to disk using the XML format (VTP)."""
    print("Saving to:", vtk_path)
    
    # Use vtkXMLPolyDataWriter instead of vtkPolyDataWriter
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(vtk_path)
    writer.SetInputData(polydata)
    
    if binary:
        # For XML files, 'binary' is the default and is handled automatically.
        # You can specify the data mode if needed:
        writer.SetDataModeToAppended() 
        writer.SetCompressorTypeToZLib() # Optional: compresses the file
    else:
        writer.SetDataModeToAscii()
        
    return writer.Write() == 1

def save_vtk(polydata, vtk_path, binary=True):
    """Saves the PolyData object to disk."""
    print("saving to ", vtk_path)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(vtk_path)
    writer.SetInputData(polydata)
    if binary:
        writer.SetFileTypeToBinary()
    else:
        writer.SetFileTypeToASCII()
    return writer.Write() == 1