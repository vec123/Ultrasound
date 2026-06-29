import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

class PointCloudGraphProcessor:
    def __init__(self, input_vtp_path):
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

    def get_knn_data(self, k=5, normalize=True):
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

# --- Example Usage ---
if __name__ == "__main__":
    INPUT_VTP = r"C:\Users\vic-b\Documents\Victors\Projects\Ultrasound\US_samples\US_samples\1\20_semanas\VTK\1-01_features.vtp"
    processor = PointCloudGraphProcessor(INPUT_VTP)

    pos, senders, receivers, rel_dists = processor.get_knn_data(k=5)
    knn_graph = processor.create_knn_graph(pos, senders, receivers)
    processor.save_for_paraview("output_knn_graph.vtp", knn_graph)
    np.savez("graph_data.npz", 
             positions=pos, 
             senders=senders, 
             receivers=receivers, 
             rel_distances=rel_dists)
    print("Graph data saved to graph_data.npz")

 