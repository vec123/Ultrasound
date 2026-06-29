from src.geometry.vtk import load_vtu, convert_vtu_to_vtp_vtk, save_vtp_mesh, filter_largest_component
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
vtu_path = os.path.join(PROJECT_ROOT, "US_samples","US_samples", "clean.vtu")
vtp_path = os.path.join(PROJECT_ROOT, "US_samples","US_samples", "clean.vtp")
print("load")
vtu = load_vtu(vtu_path)
print("convert")
vtp = convert_vtu_to_vtp_vtk(vtu)
vtp = filter_largest_component(vtp)
print("save")
save_vtp_mesh(vtp,vtp_path)