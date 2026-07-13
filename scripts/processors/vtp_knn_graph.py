import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import os
import json
from dotenv import load_dotenv

class PointCloudGraphProcessor:
    def __init__(self, input_vtp_path, json_path=None):
        self.reader = vtk.vtkXMLPolyDataReader()
        self.reader.SetFileName(input_vtp_path)
        self.reader.Update()
        self.polydata = self.reader.GetOutput()

    def get_points_as_array(self):
        """Extracts points as a NumPy array."""
        points = self.polydata.GetPoints()
        return vtk_to_numpy(points.GetData())
    
    def create_knn_graph(self, positions, senders, receivers):
            """
            Creates a VTK PolyData graph from existing NumPy arrays.
            """
            # 1. Create points from positions
            points = vtk.vtkPoints()
            for p in positions:
                points.InsertNextPoint(p)
                
            # 2. Create lines from edge list
            edges = vtk.vtkCellArray()
            for s, r in zip(senders, receivers):
                edge = vtk.vtkLine()
                edge.GetPointIds().SetId(0, int(s))
                edge.GetPointIds().SetId(1, int(r))
                edges.InsertNextCell(edge)
                
            graph = vtk.vtkPolyData()
            graph.SetPoints(points)
            graph.SetLines(edges)
            return graph

    def get_knn_data(self, k=0, normalize=True):
            # 1. Initialize lists for edges
            senders = []
            receivers = []
            rel_distances = []

            # 2. Get Positions explicitly
            # Ensure this method returns an array-like object
            positions = np.array(self.get_points_as_array(), dtype=np.float32)
            
            # 3. Build Locator
            locator = vtk.vtkKdTreePointLocator()
            locator.SetDataSet(self.polydata)
            locator.BuildLocator()

            num_points = self.polydata.GetNumberOfPoints()
            
            # 4. Iterate to find neighbors
            for i in range(num_points):
                p = self.polydata.GetPoint(i)
                result = vtk.vtkIdList()
                locator.FindClosestNPoints(k + 1, p, result)
                
                for j in range(1, result.GetNumberOfIds()):
                    neighbor_idx = result.GetId(j)
                    neighbor_p = self.polydata.GetPoint(neighbor_idx)
                    
                    dist_vec = np.array(neighbor_p) - np.array(p)
                    
                    senders.append(i)
                    receivers.append(neighbor_idx)
                    rel_distances.append(dist_vec)

            # Convert remaining lists to arrays
            senders = np.array(senders, dtype=np.int32)
            receivers = np.array(receivers, dtype=np.int32)
            rel_distances = np.array(rel_distances, dtype=np.float32)

            # 5. Optional Normalization
            if normalize:
                centroid = np.mean(positions, axis=0)
                positions -= centroid
                scale = np.std(positions) + 1e-6
                positions /= scale
                rel_distances /= scale

            return positions, senders, receivers, rel_distances
    
    def save_for_paraview(self, output_path, data_object):
        """Writes the object to a file readable by ParaView."""
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(output_path)
        writer.SetInputData(data_object)
        writer.Write()
        print(f"Successfully saved to {output_path}")


class PointRadiusProcessor(PointCloudGraphProcessor):
    def __init__(self, input_vtp_path, json_path=None):
        super().__init__(input_vtp_path)

        if json_path is not None:
            with open(json_path, 'r') as f:
                self.reference_points = json.load(f)
            
        self.locator = vtk.vtkKdTreePointLocator()
        self.locator.SetDataSet(self.polydata)
        self.locator.BuildLocator()

    def extract_all_regions(self, radius, output_dir="subset"):
        # 1. Create the directory ONCE before the loop
        os.makedirs(output_dir, exist_ok=True)
        
        for key in self.reference_points.keys():
            # 2. Use a variable for the file path
            file_name = f"{key}.vtp"
            full_file_path = os.path.join(output_dir, file_name)
            # 3. Pass that specific file path to your method
            self.extract_and_save_subgraph(key, radius, full_file_path)

    def extract_and_save_subgraph(self, key, radius, output_path):
        center = self.reference_points[key]
        
        # 1. Get IDs using the locator (remains the same)
        result_ids = vtk.vtkIdList()
        self.locator.FindPointsWithinRadius(radius, center, result_ids)
        
        # 2. Convert vtkIdList to vtkIdTypeArray
        selection_list = vtk.vtkIdTypeArray()
        selection_list.SetNumberOfComponents(1)
        selection_list.SetNumberOfTuples(result_ids.GetNumberOfIds())
        
        for i in range(result_ids.GetNumberOfIds()):
            selection_list.SetValue(i, result_ids.GetId(i))
        
        # 3. Configure the selection node
        selection_node = vtk.vtkSelectionNode()
        selection_node.SetFieldType(vtk.vtkSelectionNode.POINT)
        selection_node.SetContentType(vtk.vtkSelectionNode.INDICES)
        
        # Now we pass the vtkIdTypeArray
        selection_node.SetSelectionList(selection_list)
        
        # ... rest of your existing logic ...
        selection = vtk.vtkSelection()
        selection.AddNode(selection_node)
        
        extract = vtk.vtkExtractSelection()
        extract.SetInputData(0, self.polydata)
        extract.SetInputData(1, selection)
        
        # --- FIX: Convert UnstructuredGrid to PolyData ---
        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputConnection(extract.GetOutputPort())
        surface_filter.Update()
        
        self.save_for_paraview(output_path, surface_filter.GetOutput())


class PointSamplerProcessor(PointRadiusProcessor):
    def sample_points(self, n_samples, method='uniform'):
        """
        Samples N vertex indices from the polydata.
        """
        num_points = self.polydata.GetNumberOfPoints()
        indices = np.arange(num_points)
        
        if method == 'uniform':
            return np.random.choice(indices, n_samples, replace=False)
        
        elif method == 'fps':
            # Farthest Point Sampling
            sampled_indices = [np.random.randint(0, num_points)]
            # Get all points as numpy for distance calculations
            pts = vtk.util.numpy_support.vtk_to_numpy(self.polydata.GetPoints().GetData())
            
            distances = np.full(num_points, np.inf)
            for _ in range(n_samples - 1):
                last_idx = sampled_indices[-1]
                # Update distances to the newly added point
                dist_to_last = np.linalg.norm(pts - pts[last_idx], axis=1)
                distances = np.minimum(distances, dist_to_last)
                # Select the point with the maximum distance
                sampled_indices.append(np.argmax(distances))
            return np.array(sampled_indices)
        
        return indices[:n_samples]

    def extract_patches(self, n_samples, radius, output_dir, method='uniform', normalize=False):
        """
        Samples N locations and saves patches. 
        If normalize is True, shifts the patch to origin and scales by radius.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        sampled_indices = self.sample_points(n_samples, method=method)
        
        for i, idx in enumerate(sampled_indices):
            center = np.array(self.polydata.GetPoint(idx))
            norm_file_name = f"patch_{i:03d}_{'norm'}.vtp"
            norm_path = os.path.join(output_dir, norm_file_name)
            
            raw_file_name = f"patch_{i:03d}_{'raw'}.vtp"
            raw_path = os.path.join(output_dir, raw_file_name)
            # Use the internal logic to extract raw points
            self.reference_points = {f"patch_{i}": center}
            
            # Get the extracted polydata
            patch_data = self._get_subgraph_object(f"patch_{i}", radius)
            self.save_for_paraview(raw_path, patch_data)

            if normalize and patch_data:
                # 1. Get points as numpy array
                points = vtk_to_numpy(patch_data.GetPoints().GetData()).astype(np.float32)
                
                # 2. Calculate the Center of Mass (centroid) of the patch
                centroid = np.mean(points, axis=0)
                
                # 3. Center: Subtract the centroid so CoM moves to (0,0,0)
                points -= centroid
                
                # 4. Normalize: Scale by the radius
                if radius > 0:
                    points /= radius
                
                # 5. Update the VTK object
                new_points = vtk.vtkPoints()
                for p in points:
                    new_points.InsertNextPoint(p)
                patch_data.SetPoints(new_points)
            
                self.save_for_paraview(norm_path, patch_data)

    def _get_subgraph_object(self, key, radius):
        """Helper to return the vtkPolyData of a selection without saving."""
        center = self.reference_points[key]
        result_ids = vtk.vtkIdList()
        self.locator.FindPointsWithinRadius(radius, center, result_ids)
        
        selection_list = vtk.vtkIdTypeArray()
        selection_list.SetNumberOfTuples(result_ids.GetNumberOfIds())
        for i in range(result_ids.GetNumberOfIds()):
            selection_list.SetValue(i, result_ids.GetId(i))
            
        selection_node = vtk.vtkSelectionNode()
        selection_node.SetFieldType(vtk.vtkSelectionNode.POINT)
        selection_node.SetContentType(vtk.vtkSelectionNode.INDICES)
        selection_node.SetSelectionList(selection_list)
        
        selection = vtk.vtkSelection()
        selection.AddNode(selection_node)
        
        extract = vtk.vtkExtractSelection()
        extract.SetInputData(0, self.polydata)
        extract.SetInputData(1, selection)
        
        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputConnection(extract.GetOutputPort())
        surface_filter.Update()
        return surface_filter.GetOutput()

# --- Example Usage ---
if __name__ == "__main__":
    load_dotenv()
    ROOT = os.environ.get("PROJECT_ROOT")
    DATA_PATH = os.path.join(ROOT, "US_samples", "US_samples_hr")
    INPUT_VTP = os.path.join(DATA_PATH, "1", "20_semanas", "VTK", "1-02.vtp")
    INPUT_JSON = os.path.join(DATA_PATH, "1", "20_semanas", "VTK" ,
                             "1-02.json")
    SUBPARTS_DIR = os.path.join(DATA_PATH, "1", "20_semanas", "VTK",
                             "subparts" , "2")
    PATCHES_DIR = os.path.join(DATA_PATH, "1", "20_semanas", "VTK",
                             "patches" , "2")
    NORM_PATCHES_DIR = os.path.join(DATA_PATH, "1", "20_semanas", "VTK",
                             "patches" , "2")
    processor = PointCloudGraphProcessor(INPUT_VTP)

    pos, senders, receivers, rel_dists = processor.get_knn_data(k=0)
    knn_graph = processor.create_knn_graph(pos, senders, receivers)
    processor.save_for_paraview("output_knn_graph.vtp", knn_graph)
    np.savez("graph_data.npz", 
             positions=pos, 
             senders=senders, 
             receivers=receivers, 
             rel_distances=rel_dists)
    print("Graph data saved to graph_data.npz")


    processor = PointRadiusProcessor(INPUT_VTP, INPUT_JSON)
    processor.extract_all_regions(radius=5.0, output_dir = SUBPARTS_DIR)
 
    processor = PointSamplerProcessor(INPUT_VTP, None) # JSON not needed for sampling
    
    #processor.extract_patches(n_samples=100, radius=5.0, output_dir=PATCHES_DIR, method='fps')
    processor.extract_patches(n_samples=100, radius=8.0, output_dir=NORM_PATCHES_DIR, method='fps', normalize = True)