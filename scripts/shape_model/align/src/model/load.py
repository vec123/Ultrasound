import scipy.io as sio
import numpy as np


def load_mat_ssm(path, normalize = True):
    # --- LOAD DATA ---
    mat_data = sio.loadmat(path, squeeze_me=True)
    FaceModel = mat_data['BabyFaceModel']

    refShape = FaceModel['refShape'].item()
    pctVar_per_eigen = FaceModel['pctVar_per_eigen'].item()
    eigenValues = FaceModel['eigenValues'].item()
    eigenFunctions = FaceModel['eigenFunctions'].item()
    triang = FaceModel['triang'].item()
    meanDeformation= FaceModel['meanDeformation'].item()
    lmks_vertsIND = FaceModel['landmark_verts'].item() - 1
    landmark_names = [str(n).strip() for n in FaceModel['landmark_names'].item()]
    
    #  Reconstruct full mean
    shapeMU = refShape.flatten(order='F')
    mean = shapeMU + meanDeformation
    
    if normalize:
        # Normalize the Mean
        mean_reshaped = mean.reshape(-1, 3)
        centroid = mean_reshaped.mean(axis=0)
        centered_mean = mean_reshaped - centroid
        
        landmark_data = {"verts ": lmks_vertsIND, "names ": landmark_names}
    
        # Calculate scale: Root Mean Square distance to centroid
        rms_dist = np.sqrt(np.mean(np.sum(centered_mean**2, axis=1)))
        scale = 1.0 / rms_dist
        
        normalized_mean = (centered_mean * scale).flatten()
        
        # 3. Apply normalization to Eigenfunctions
        # The eigenfunctions define shape change, so they must be scaled by the same factor
        # but do NOT get translated by the centroid.
        normalized_eigenfunctions = eigenFunctions * scale
        mean = normalized_mean
        eigenFunctions = normalized_eigenfunctions

    return mean, eigenFunctions, eigenValues, landmark_data #centroid, scale

""" 
def load_mat_ssm_(path):

    # --- LOAD DATA ---
    mat_data = sio.loadmat(path, squeeze_me=True)
    FaceModel = mat_data['BabyFaceModel']

    refShape = FaceModel['refShape'].item()
    pctVar_per_eigen = FaceModel['pctVar_per_eigen'].item()
    eigenValues = FaceModel['eigenValues'].item()
    eigenFunctions = FaceModel['eigenFunctions'].item()
    triang = FaceModel['triang'].item()
    meanDeformation= FaceModel['meanDeformation'].item()
    lmks_vertsIND = FaceModel['landmark_verts'].item() - 1
    landmark_names = [str(n).strip() for n in FaceModel['landmark_names'].item()]

    landmark_data = {"verts ": lmks_vertsIND,
                 "names ": landmark_names}
    
    shapeMU = refShape.flatten(order='F')
    mean = shapeMU + meanDeformation

    return mean, eigenFunctions, eigenValues, landmark_data
"""