import SimpleITK as sitk
import vtk
import numpy as np
from vtk.util import numpy_support


def convert_single_nrrd(nrrd_path, contour_value=None):
    """
    Converts a single NRRD file to a VTP file.
    """
    # 1. Load the NRRD file with SimpleITK
    sitk_image = sitk.ReadImage(nrrd_path)
    
    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()
    np_array = sitk.GetArrayFromImage(sitk_image)
    
    # 2. Convert to vtkImageData
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(np_array.shape[2], np_array.shape[1], np_array.shape[0])
    vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
    vtk_image.SetOrigin(origin[0], origin[1], origin[2])
    
    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=np_array.ravel(), 
        deep=True, 
        array_type=numpy_support.get_vtk_array_type(np_array.dtype)
    )
    vtk_data_array.SetName("Scalars")
    vtk_image.GetPointData().SetScalars(vtk_data_array)
    
    # 3. Convert vtkImageData to vtkPolyData (Geometry generation)
    if contour_value is not None:
        contour_filter = vtk.vtkMarchingCubes()
        contour_filter.SetInputData(vtk_image)
        contour_filter.SetValue(0, contour_value)
        contour_filter.Update()
        poly_data = contour_filter.GetOutput()
    else:
        geometry_filter = vtk.vtkGeometryFilter()
        geometry_filter.SetInputData(vtk_image)
        geometry_filter.Update()
        poly_data = geometry_filter.GetOutput()


    return poly_data