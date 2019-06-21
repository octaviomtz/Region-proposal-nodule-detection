
# coding: utf-8

# # Modification
# 
# **DEBUG**
# 1. we temporaly remove the padding function (after loading data), it should be restored
# 
# **v3.6 complement of the preprocessing that should be run after inpainting ** 
# 1. We resample along the slices
# 1. We reproduce the inpainting steps (where small cubes where obtained) to obtain the right coordinates of the nodules
# 1. These coordinates are obtained from the pylidc and with the info
# 
# **dicom full-preprocessing (misc1) v3 - interpolate only 2axes to inpaint**
# 1. We resample along the vertical and horizontal axes but we don't resample along the slices. This creates a smaller volume to apply inpainting. The slices axis has to be resamples later
# 1. We dilate the union of the segmentations
# 1. We dilate the lungs mask with kernel=1 (it was 5)


# Compared to the previous version (v2), this script removes the scans with bad
# slices (>2.5mm or inconsistency between spacing and thickness)

import os # module for interfacing with the os
import numpy as np # numpy for arrays etc
import pandas as pd # module for creating and querying data tables (databases) efficiently
import pylidc as pl # module for handling the LIDC dataset
import matplotlib.pyplot as plt # plotting utilities
import matplotlib.patches as patches
import scipy.ndimage # ~
import scipy.sparse
import scipy
from preprocessing.preprocess_functions import *
#from utils_LIDC.utils_LIDC import *
from pylidc.utils import consensus
from skimage.morphology import ball, dilation
import sys



sys.path.insert(0,'../python_custom_functions')
from inpainting_nodules_functions import *



from scipy import sparse
from tqdm import tqdm_notebook



LIDC_PATH = '/data/datasets/LIDC-IDRI/' # original LIDC data
# annotations = pd.read_csv('/data/datasets/LIDC-IDRI/annotations.csv')
LIDC_IDs = os.listdir(f'{LIDC_PATH}LIDC-IDRI')
LIDC_IDs = [i for i in LIDC_IDs if 'LIDC' in i]
LIDC_IDs = np.sort(LIDC_IDs)

# output path
#path_dest = f'/data/OMM/Datasets/LIDC_other_formats/LIDC_preprocessed_3D v5 - save pylidc chars only/v17v2/' 
path_dest = f'/data/OMM/Datasets/LIDC_other_formats/LIDC_preprocessed_3D v5 - save pylidc chars only/v17v3/' 
path_data_alreadyprocessed = '/data/OMM/Datasets/LIDC_other_formats/LIDC_preprocessed_3D v4 - inpaint before preprocess/'
#path_already_inpainted = '/data/OMM/project results/Feb 5 19 - Deep image prior/dip results all 17/arrays/'
#path_already_inpainted = '/data/OMM/project results/Feb 5 19 - Deep image prior/dip results all 17/v17v2_merged_clusters/arrays/'
path_already_inpainted = '/data/OMM/project results/Feb 5 19 - Deep image prior/dip results all 17v3/arrays/'
if not os.path.exists(path_dest): os.makedirs(path_dest)


# ## Functions


def nodule_coords_in_small_resampled_versions(df, resampling_ratio, min_box_x, min_box_y, min_box_channels):
    '''
    Get the coordinates of the nodules in the smaller resampled volumes.
    This is done to be able to link each nodule to their pylidc labels
    We need to get into account the resampling ratio and the number of voxels used during the
    "Find the minimum box that contain the lungs" of the "read_slices3D_v3" function
    '''
    pd.options.mode.chained_assignment = None
    df['small_coordsZ']=df['lidc_coordZ'].values * resampling_ratio[0] - np.min(min_box_channels)
    df['small_coordsX']=df['lidc_coordX'].values * resampling_ratio[1] - np.min(min_box_x)
    df['small_coordsY']=df['lidc_coordY'].values * resampling_ratio[2] - np.min(min_box_y)
    return df



def nodule_coords_in_small_resampled_versions2(df, resampling_ratio, min_box_x, min_box_y, min_box_z,
                                              slice_middle, xmed_1, ymed_1, xmed_2, ymed_2):
    '''
    Get the coordinates of the nodules in the smaller resampled volumes.
    This is done to be able to link each nodule to their pylidc labels
    We need to get into account the resampling ratio and the number of voxels used during the
    "Find the minimum box that contain the lungs" of the "read_slices3D_v3" function
    '''
    pd.options.mode.chained_assignment = None
    # Transform the original coords to small cube coords
    COORDZ = (np.mean(df['lidc_coordZ'].values) * resampling_ratio[0]) - np.min(min_box_z)
    COORDX = (np.mean(df['lidc_coordX'].values) * resampling_ratio[1]) - np.min(min_box_x)
    COORDY = (np.mean(df['lidc_coordY'].values) * resampling_ratio[2]) - np.min(min_box_y)
    coords_in_small_cube = np.asarray(COORDZ, COORDX, COORDY)
    # Find if nodule is closer to left or right nodule
    # MAYBE JUST COMPARE THE Z COORD AGAINST THE SLICE MIDDLE
    #coords_center_cube1 = np.asarray(slice_middle, xmed_1, ymed_1)
    #coords_center_cube2 = np.asarray(slice_middle, xmed_2, ymed_2)
    #dist1 = np.linalg.norm(coords_in_small_cube - coords_center_cube1)
    #dist2 = np.linalg.norm(coords_in_small_cube - coords_center_cube2)
    #COMPARE ONLY THE Y DIRECTION
    dist1=np.abs(COORDY-ymed_1)
    dist2=np.abs(COORDY-ymed_2)
    if dist1<dist2: 
        coord_adj_Z = ((df['lidc_coordZ'].values * resampling_ratio[0]) - np.min(min_box_z)) - c_zmin1
        coord_adj_X = ((df['lidc_coordX'].values * resampling_ratio[1]) - np.min(min_box_x)) - c_xmin1
        coord_adj_Y = ((df['lidc_coordY'].values * resampling_ratio[2]) - np.min(min_box_y)) - c_ymin1
        nodule_in_block = 1
        #print('1', coord_adj_Z, coord_adj_X, coord_adj_Y)
        
    else: 
        coord_adj_Z = ((df['lidc_coordZ'].values * resampling_ratio[0]) - np.min(min_box_z)) - c_zmin2
        coord_adj_X = ((df['lidc_coordX'].values * resampling_ratio[1]) - np.min(min_box_x)) - c_xmin2
        coord_adj_Y = ((df['lidc_coordY'].values * resampling_ratio[2]) - np.min(min_box_y)) - c_ymin2
        nodule_in_block = 2
        #print('2', coord_adj_Z, coord_adj_X, coord_adj_Y)
    
    df['small_coordsZ']= coord_adj_Z
    df['small_coordsX']= coord_adj_X
    df['small_coordsY']= coord_adj_Y
    df['nodule_in_block'] = nodule_in_block 
    return df, coord_adj_Z, coord_adj_X, coord_adj_Y



def resample_grid_except_slices(image, spacing, new_spacing=[1,1,1],method='linear'):
    '''resample along the vertical and horizontal axes but don't resample along the slices.
    This creates a smaller volume to apply inpainting. The slices axis has to be resamples later'''    
    x, y, z = [spacing[k] * np.arange(image.shape[k]) for k in range(3)]  # original grid in mm
    x = np.arange(np.shape(image)[0]) # we dont interpolate in x direction (slices)
    f = scipy.interpolate.RegularGridInterpolator((x, y, z), image)#, method='linear')    # interpolator
#    print('Interpolating')

    dx, dy, dz = new_spacing    # new step sizes
    new_grid = np.mgrid[0:x[-1]+1:dx, 0:y[-1]:dy, 0:z[-1]:dz] # a '+1' is added
    new_grid = np.moveaxis(new_grid, (0, 1, 2, 3), (3, 0, 1, 2))  # reorder axes for evaluation
    imageOut = f(new_grid)
    
    # convert back to the same type as input (if it was an int, round first!)
    dataType = image.dtype
    if np.issubdtype(image[0,0,0],np.signedinteger) or np.issubdtype(image[0,0,0],np.unsignedinteger):
        imageOut = np.round(imageOut)
        
    imageOut = imageOut.astype(dataType)
    
    return imageOut, new_spacing 



def resample_grid_slices(image, spacing, new_spacing=[1,1,1],method='linear'):
    '''DO NOT resample along the vertical and horizontal axes, ONLY resample along the slices.
    This is done because before inpainting we applied resampling only to the other two dimensions.
    (check resample_grid_except_slices)'''    
    x, y, z = [spacing[k] * np.arange(image.shape[k]) for k in range(3)]  # original grid in mm
    # we only interpolate in x direction (slices)
    y = np.arange(np.shape(image)[1])
    z = np.arange(np.shape(image)[2]) 
    f = scipy.interpolate.RegularGridInterpolator((x, y, z), image)#, method='linear')    # interpolator
#    print('Interpolating')

    dx, dy, dz = new_spacing    # new step sizes
    new_grid = np.mgrid[0:x[-1]:dx, 0:y[-1]+1:dy, 0:z[-1]+1:dz] # a '+1' is added
    new_grid = np.moveaxis(new_grid, (0, 1, 2, 3), (3, 0, 1, 2))  # reorder axes for evaluation
    imageOut = f(new_grid)
    
    # convert back to the same type as input (if it was an int, round first!)
    dataType = image.dtype
    if np.issubdtype(image[0,0,0],np.signedinteger) or np.issubdtype(image[0,0,0],np.unsignedinteger):
        imageOut = np.round(imageOut)
        
    imageOut = imageOut.astype(dataType)
    
    return imageOut, new_spacing 



def plot_resampled_block(df_coords_adjusted_, mask_resampled_, block_name_, pid_):
    df_block = df_coords_adjusted_.loc[df_coords_adjusted_['nodule_in_block']==int(block_name_[-1])]
    # Get the values from the DF
    df_number_of_nodules = np.unique(df_block.cluster_id.values)
    for nn in df_number_of_nodules:
        df_nodule_freeze=df_coords_adjusted_.loc[df_coords_adjusted_['cluster_id']==nn] 
        zz = int(np.mean(df_nodule_freeze['small_coordsZ_resampled']))
        xx = int(np.mean(df_nodule_freeze['small_coordsX']))
        yy = int(np.mean(df_nodule_freeze['small_coordsY']))
        # Get the resampled block
        rect = patches.Rectangle((np.maximum(yy-20,0),np.maximum(xx-20,0)),40,40,linewidth=1,edgecolor='r',facecolor='none')
        fig, ax = plt.subplots(1,1)
        ax.set_title(f'{pid_}_{block_name_}\n{zz,xx,yy}')
        ax.imshow(mask_resampled_[zz])
        ax.axis('off')
        ax.add_patch(rect)


# In[24]:


def nodule_coords_in_small_resampled_versions4(df, resampling_ratio, min_box_x, min_box_y, min_box_z,
                                              slice_middle, ymed_1, ymed_2, box_coords1, box_coords2, 
                                               nodule_centers_from_image):
    '''
    Get the coordinates of the nodules in the smaller resampled volumes.
    This is done to be able to link each nodule to their pylidc labels
    We need to get into account the resampling ratio and the number of voxels used during the
    "Find the minimum box that contain the lungs" of the "read_slices3D_v3" function
    v4 works with boxes with multiple nodules:
    We use "nodule_centers_from_image" to get the center(s) of the nodule(s) from the boxes (from the images 
    using ndimage.label)
    We pass to this function the coordinates of the (possible different) boxes (in e.g. box_coords2)
    We also pass a df CONTAINING ONLY THE INDICES OF THE NODULES inside the box. 
    To do this we check the NAME of the box.
    Then, for each possible combination of box coords and nodule coords we compute the coords (from the 
    box perspective). If the distance from the computed coords and the coords from the images (using ndimage.label)
    along the x and y axes is small then we keep the new calculated coord. We only use x and y because we will
    resample along the z axis
    '''
    
    pd.options.mode.chained_assignment = None
    # Transform the original coords to small cube coords
    COORDZ = (np.mean(df['lidc_coordZ'].values) * resampling_ratio[0]) - np.min(min_box_z)
    COORDX = (np.mean(df['lidc_coordX'].values) * resampling_ratio[1]) - np.min(min_box_x)
    COORDY = (np.mean(df['lidc_coordY'].values) * resampling_ratio[2]) - np.min(min_box_y)
    coords_in_small_cube = np.asarray(COORDZ, COORDX, COORDY)
    # Find if nodule is closer to left or right nodule
    
    #COMPARE ONLY THE Y DIRECTION
    
    dist1=np.abs(COORDY-ymed_1)
    dist2=np.abs(COORDY-ymed_2)
    
    if dist1<dist2:
        coords_found = False
        for box_coords1_one_box in box_coords1:# for each left box
            c_zmin1, c_zmax1, c_xmin1, c_xmax1, c_ymin1, c_ymax1 = box_coords1_one_box
            if coords_found: break
            coord_adj_Z = (np.mean(df['lidc_coordZ'].values * resampling_ratio[0]) - np.min(min_box_z)) - c_zmin1
            coord_adj_X = (np.mean(df['lidc_coordX'].values * resampling_ratio[1]) - np.min(min_box_x)) - c_xmin1
            coord_adj_Y = (np.mean(df['lidc_coordY'].values * resampling_ratio[2]) - np.min(min_box_y)) - c_ymin1
            nodule_in_block = 1
            for center_one_nodule in nodule_centers_from_image:
                dist_to_ndl = np.sum(np.abs(np.asarray(center_one_nodule)[1:] - np.asarray([coord_adj_Z, coord_adj_X, coord_adj_Y])[1:]))
                if dist_to_ndl < 5:
                    coord_adj_Z_correct_ndl = coord_adj_Z
                    coord_adj_X_correct_ndl = coord_adj_X
                    coord_adj_Y_correct_ndl = coord_adj_Y
                    coords_found = True
                    
                    df['small_coordsZ']= coord_adj_Z_correct_ndl
                    df['small_coordsX']= coord_adj_X_correct_ndl
                    df['small_coordsY']= coord_adj_Y_correct_ndl
                    df['nodule_in_block'] = nodule_in_block 
                    return [df, coord_adj_Z, coord_adj_X, coord_adj_Y, coords_found], coords_found
        return [0], coords_found
            
        
    else: 
        coords_found = False
        for box_coords2_one_box in box_coords2: # for each right box
            c_zmin2, c_zmax2, c_xmin2, c_xmax2, c_ymin2, c_ymax2 = box_coords2_one_box
            if coords_found: break
            coord_adj_Z = (np.mean(df['lidc_coordZ'].values * resampling_ratio[0]) - np.min(min_box_z)) - c_zmin2
            coord_adj_X = (np.mean(df['lidc_coordX'].values * resampling_ratio[1]) - np.min(min_box_x)) - c_xmin2
            coord_adj_Y = (np.mean(df['lidc_coordY'].values * resampling_ratio[2]) - np.min(min_box_y)) - c_ymin2
            nodule_in_block = 2
            for center_one_nodule in nodule_centers_from_image: # for each nodule in the box                
                dist_to_ndl = np.sum(np.abs(np.asarray(center_one_nodule)[1:] - np.asarray([coord_adj_Z, coord_adj_X, coord_adj_Y])[1:]))                            
                if dist_to_ndl < 5:
                    coord_adj_Z_correct_ndl = coord_adj_Z
                    coord_adj_X_correct_ndl = coord_adj_X
                    coord_adj_Y_correct_ndl = coord_adj_Y
                    coords_found = True
            
                    df['small_coordsZ']= coord_adj_Z_correct_ndl
                    df['small_coordsX']= coord_adj_X_correct_ndl
                    df['small_coordsY']= coord_adj_Y_correct_ndl
                    df['nodule_in_block'] = nodule_in_block 
                    return [df, coord_adj_Z, coord_adj_X, coord_adj_Y, coords_found], coords_found
        return [0], coords_found




def nodule_coords_in_small_resampled_versions4_5(df, resampling_ratio, min_box_x, min_box_y, min_box_z,
                                              slice_middle, ymed_1, ymed_2, box_coords1, box_coords2, 
                                               nodule_centers_from_image):
    '''
    Get the coordinates of the nodules in the smaller resampled volumes.
    This is done to be able to link each nodule to their pylidc labels
    We need to get into account the resampling ratio and the number of voxels used during the
    "Find the minimum box that contain the lungs" of the "read_slices3D_v3" function
    v4 works with boxes with multiple nodules:
    We use "nodule_centers_from_image" to get the center(s) of the nodule(s) from the boxes (from the images 
    using ndimage.label)
    We pass to this function the coordinates of the (possible different) boxes (in e.g. box_coords2)
    We also pass a df CONTAINING ONLY THE INDICES OF THE NODULES inside the box. 
    To do this we check the NAME of the box.
    Then, for each possible combination of box coords and nodule coords we compute the coords (from the 
    box perspective). If the distance from the computed coords and the coords from the images (using ndimage.label)
    along the x and y axes is small then we keep the new calculated coord. We only use x and y because we will
    resample along the z axis
    INPUTS:
    - df: the dataframe of a SINGLE nodule of a SINGLE patient
    '''
    
    pd.options.mode.chained_assignment = None
    # Transform the original coords to small cube coords
    COORDZ = (np.mean(df['lidc_coordZ'].values) * resampling_ratio[0]) - np.min(min_box_z)
    COORDX = (np.mean(df['lidc_coordX'].values) * resampling_ratio[1]) - np.min(min_box_x)
    COORDY = (np.mean(df['lidc_coordY'].values) * resampling_ratio[2]) - np.min(min_box_y)
    coords_in_small_cube = np.asarray(COORDZ, COORDX, COORDY)
    # Find if nodule is closer to left or right nodule
    
    #COMPARE ONLY THE Y DIRECTION
    
    dist1=np.abs(COORDY-ymed_1)
    dist2=np.abs(COORDY-ymed_2)
    
    if dist1<dist2: # for each left box 
        coords_found = False 
        for box_coords1_one_box in box_coords1: # compute the relative coordinates of the nodule in the df
            c_zmin1, c_zmax1, c_xmin1, c_xmax1, c_ymin1, c_ymax1 = box_coords1_one_box
            if coords_found: break
            coord_adj_Z = (np.mean(df['lidc_coordZ'].values * resampling_ratio[0]) - np.min(min_box_z)) - c_zmin1
            coord_adj_X = (np.mean(df['lidc_coordX'].values * resampling_ratio[1]) - np.min(min_box_x)) - c_xmin1
            coord_adj_Y = (np.mean(df['lidc_coordY'].values * resampling_ratio[2]) - np.min(min_box_y)) - c_ymin1
            nodule_in_block = 1
            dist_min = 100
            for center_one_nodule in nodule_centers_from_image: # for each nodule in the box compute the difference to the box relative coordinates
                dist_to_ndl = np.sum(np.abs(np.asarray(center_one_nodule)[1:] - np.asarray([coord_adj_Z, coord_adj_X, coord_adj_Y])[1:]))
                if dist_to_ndl < dist_min: # we take the closest coordinates
                    dist_min = dist_to_ndl 
                    if dist_to_ndl < 5: # if the closest coordinates are close
                        coord_adj_Z_correct_ndl = coord_adj_Z
                        coord_adj_X_correct_ndl = coord_adj_X
                        coord_adj_Y_correct_ndl = coord_adj_Y
                        coords_found = True
                    
                        df['small_coordsZ']= coord_adj_Z_correct_ndl
                        df['small_coordsX']= coord_adj_X_correct_ndl
                        df['small_coordsY']= coord_adj_Y_correct_ndl
                        df['nodule_in_block'] = nodule_in_block 
                        # then return the coordinates
                        return [df, coord_adj_Z, coord_adj_X, coord_adj_Y, coords_found], coords_found, center_one_nodule, [coord_adj_Z_correct_ndl, coord_adj_X_correct_ndl, coord_adj_Y_correct_ndl]
        return [0], coords_found, '', ''
            
        
    else: # for each right box
        coords_found = False
        for box_coords2_one_box in box_coords2: # compute the relative coordinates of the nodule in the df
            c_zmin2, c_zmax2, c_xmin2, c_xmax2, c_ymin2, c_ymax2 = box_coords2_one_box
            if coords_found: break
            coord_adj_Z = (np.mean(df['lidc_coordZ'].values * resampling_ratio[0]) - np.min(min_box_z)) - c_zmin2
            coord_adj_X = (np.mean(df['lidc_coordX'].values * resampling_ratio[1]) - np.min(min_box_x)) - c_xmin2
            coord_adj_Y = (np.mean(df['lidc_coordY'].values * resampling_ratio[2]) - np.min(min_box_y)) - c_ymin2
            nodule_in_block = 2
            dist_min = 100
            for center_one_nodule in nodule_centers_from_image: # for each nodule in the box compute the difference to the box relative coordinates               
                dist_to_ndl = np.sum(np.abs(np.asarray(center_one_nodule)[1:] - np.asarray([coord_adj_Z, coord_adj_X, coord_adj_Y])[1:]))                            
                if dist_to_ndl < dist_min: # we take the closest coordinates
                    dist_min = dist_to_ndl 
                    if dist_to_ndl < 5: # if the closest coordinates are close
                        coord_adj_Z_correct_ndl = coord_adj_Z
                        coord_adj_X_correct_ndl = coord_adj_X
                        coord_adj_Y_correct_ndl = coord_adj_Y
                        coords_found = True

                        df['small_coordsZ']= coord_adj_Z_correct_ndl
                        df['small_coordsX']= coord_adj_X_correct_ndl
                        df['small_coordsY']= coord_adj_Y_correct_ndl
                        df['nodule_in_block'] = nodule_in_block 
                        # then we return the coordinates
                        return [df, coord_adj_Z, coord_adj_X, coord_adj_Y, coords_found], coords_found, center_one_nodule, [coord_adj_Z_correct_ndl, coord_adj_X_correct_ndl, coord_adj_Y_correct_ndl]
        return [0], coords_found, '', ''




def get_median_coords_from_image(nodules_mask):
    '''For each nodule in the box nodules' mask we return their centers.
    This is done because there might be different boxes that capture different sets of nodules. 
    Then, in order to match the coordinates (in the box) of each nodule to their corresponding pylidc coords
    we are going to compare the box coordinates options calculated to the actual nodule coordinates in the box'''
    centers = []
    labeled, nr_items = ndimage.label(nodules_mask)
    for i in np.arange(1, nr_items+1):
        z,x,y = np.where(labeled==i)
        zz =  int(np.median(z))
        xx =  int(np.median(x))
        yy =  int(np.median(y))
        centers.append([zz,xx,yy])
    return centers


# ## Main

names_checkup = []
coord_mask_all = []
coord_calc_all = []
df = pd.read_csv('/data/datasets/LIDC-IDRI/annotations.csv')
ids_already_inpainted =  os.listdir(f'{path_already_inpainted}last')
ids_already_inpainted = np.sort(ids_already_inpainted)

for idx, k in enumerate(ids_already_inpainted):
    if idx < 111: continue
    print(idx)
    # errors in LIDC-IDRI-0052, LIDC-IDRI-0065, LIDC-IDRI-0068
#     if idx<2:continue
#     if idx == 5: break
#     if idx<=15:continue
#     if idx>20:continue
    name_original = k
    k = k.split('_block')[0]
    df_patient = df.loc[df['patientid']==int(k[-4:])] 
    pid = k

    # query the LIDC images with patient_id = pid 
    # HERE WE JUST USE THE FIRST ONE!!
    idx_scan = 0 

    # get the scan object for this scan
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid)[idx_scan] 

    # here we can reject according to any criteria we like
    thickSlice = (scan.slice_thickness > 3) | (scan.slice_spacing > 3)
    missingSlices = len(np.unique(np.round(100*np.diff(scan.slice_zvals)))) != 1
    if (thickSlice):
        # we want to reject this scan/patient
        print('Undesirable slice characteristics, rejecting')
        listOfRejectedPatients.append(pid)
        #continue
        raise ValueError('Undesirable slice characteristics, rejecting')
    elif (missingSlices):
        print('Missing slices, rejecting')
        listOfRejectedPatients.append(pid)
        #continue
        raise ValueError('Missing slices, rejecting')
    else:
        pass
        # we will use this scan
        # listOfUsedPatients.append(pid)
        #continue # this lets us quickly check the outcome of the selection

    # V3.6 REPEAT THE STEPS FROM INPAINTING TO GET THE TRANSFORMED COORDINATES
    path_data = path_data_alreadyprocessed
    # try:
    vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small, min_box_channels, min_box_x, min_box_y = read_slices3D_v3(path_data, pid);
    # except FileNotFoundError: continue
    vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small = pad_if_vol_too_small(vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small)
    slice_middle = np.shape(vol_small)[0] // 2
    xmed_1, ymed_1, xmed_2, ymed_2 = erode_and_split_mask(mask_lungs_small,slice_middle)
    coord_min_side1, coord_max_side1, coord_min_side2, coord_max_side2 = nodule_right_or_left_lung(mask_maxvol_small, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2)
    try:
        c_zmin2, c_zmax2, c_xmin2, c_xmax2, c_ymin2, c_ymax2 = box_coords_contain_masks_right_size_search(coord_max_side2, coord_min_side2, 2, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, mask_lungs_small, 1)
        c_zmin1, c_zmax1, c_xmin1, c_xmax1, c_ymin1, c_ymax1 = box_coords_contain_masks_right_size_search(coord_max_side1, coord_min_side1, 1,  slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, mask_lungs_small, 1)
    except ValueError: continue

    block1, block1_mask, block1_mask_maxvol_and_lungs, block1_mask_lungs = get_four_blocks(vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small, c_zmin1, c_zmax1, c_xmin1, c_xmax1, c_ymin1, c_ymax1)
    block2, block2_mask, block2_mask_maxvol_and_lungs, block2_mask_lungs = get_four_blocks(vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small, c_zmin2, c_zmax2, c_xmin2, c_xmax2, c_ymin2, c_ymax2) 
    # Normalize, clip and apply mask
    block1 = normalize_clip_and_mask(block1, block1_mask_lungs)
    block2 = normalize_clip_and_mask(block2, block2_mask_lungs)
    # Get those blocks where there is a nodule in
    blocks_ndl, blocks_ndl_mask, block_mask_maxvol_and_lungs, blocks_ndl_lungs, blocks_ndl_names, slice1, slice2 =  get_block_if_ndl(block1, block2, block1_mask, block2_mask, block1_mask_maxvol_and_lungs, block2_mask_maxvol_and_lungs, block1_mask_lungs, block2_mask_lungs)
    # There should be at least one nodule in one block, if there are not, try merging the nodules and
    # analyzing the obtained clusters separately
    #print(f'blocks_ndl = {np.shape(blocks_ndl)}')
    if len(blocks_ndl)==0:
        # Block1
        if c_zmin1==-1:
            block1_list, block1_mask_list, block1_mask_maxvol_and_lungs_list, block1_mask_lungs_list, clus_names1, box_coords1  = get_box_coords_per_block(coord_min_side1, coord_max_side1, 1, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small)
            #c_zmin1, c_zmax1, c_xmin1, c_xmax1, c_ymin1, c_ymax1 = box_coords1
        else:
            block1_list, block1_mask_list, block1_mask_maxvol_and_lungs_list, block1_mask_lungs_list, clus_names1, box_coords1 = [], [], [], [], -2, []
        # Block2
        if c_zmin2==-1:
            block2_list, block2_mask_list, block2_mask_maxvol_and_lungs_list, block2_mask_lungs_list, clus_names2, box_coords2  = get_box_coords_per_block(coord_min_side2, coord_max_side2, 2, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small)
            #c_zmin2, c_zmax2, c_xmin2, c_xmax2, c_ymin2, c_ymax2 = box_coords2
        else:
            block2_list, block2_mask_list, block2_mask_maxvol_and_lungs_list, block2_mask_lungs_list, clus_names2, box_coords2 = [], [], [], [], -2, []
        # Put them together
        blocks_ndl, blocks_ndl_mask, block_mask_maxvol_and_lungs, blocks_ndl_lungs, blocks_ndl_names = get_block_if_ndl_list(block1_list, block2_list, block1_mask_list, block2_mask_list, block1_mask_maxvol_and_lungs_list, block2_mask_maxvol_and_lungs_list, block1_mask_lungs_list, block2_mask_lungs_list, clus_names1, clus_names2)
    #print(f'blocks_ndl = {np.shape(blocks_ndl)}')
    #print('Loading and converting to HU')
    curr_patient_pixels, spacing_orig = custom_load_scan_to_HU(scan)

    #print('Resampling to isotropic resolution')
    pix_resampled, spacing = resample_grid_except_slices(curr_patient_pixels, spacing_orig, [1,1,1])


    # get all the annotations for this scan
    ids = [i.id for i in scan.annotations] # this gives the annotation IDs (note that they are not in order in the annotations.csv)

    # we split the df for patient pid into the part for just this scan
    df_patient_partX = df_patient.loc[df_patient.annotation_id.isin(ids)]
    unique_nodules = np.unique(df_patient_partX['cluster_id'].values)
    #nods = scan.cluster_annotations() # get the annotations for all nodules in this scan

    # RESAMPLE ALONG THE SLICES THE ALREADY INPAINTED IMAGES    
    #     df_coords_adjusted.to_csv(f'{path_dest}pylidc_characteristics/{pid}.csv', index=False)
    #     del block1, block2, vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small
    df_coords_adjusted_all = pd.DataFrame()
    for id_block, (block, block_mask, block_maxvol_and_lungs, block_lungs, block_name) in enumerate(zip(blocks_ndl, blocks_ndl_mask, block_mask_maxvol_and_lungs, blocks_ndl_lungs, blocks_ndl_names)):
        print(f'{block_name}   --------')
    #     if id_block==1: break
        # Get the inpainted and original image and the mask
        try:
            last = np.load(f'{path_already_inpainted}last/{pid}_{block_name}.npy')
        except FileNotFoundError: continue
        last = np.squeeze(last)
        orig = np.load(f'{path_already_inpainted}orig/{pid}_{block_name}.npy')
        orig = np.squeeze(orig)
        mask = np.load(f'{path_already_inpainted}masks nodules/{pid}_{block_name}.npz')
        mask = mask.f.arr_0
        mask_lungs = np.load(f'{path_already_inpainted}masks lungs/{pid}_{block_name}.npz')
        mask_lungs = mask_lungs.f.arr_0
    #     except FileNotFoundError: continue

        last_resampled, spacing = resample_grid_slices(last, spacing_orig, [1,1,1])
        orig_resampled, spacing = resample_grid_slices(orig, spacing_orig, [1,1,1])
        mask_resampled, spacing = resample_grid_slices(mask, spacing_orig, [1,1,1])
        mask_lungs_resampled, spacing = resample_grid_slices(mask_lungs, spacing_orig, [1,1,1])

        nodule_centers_from_image = get_median_coords_from_image(mask_resampled)

        #nodules_names_in_block = list(block_name.split('_')[-1])
        #nodules_names_in_block = [int(i) for i in nodules_names_in_block]
        possible_nodules = np.unique(df_patient_partX['cluster_id'].values)
        for idx_unique, unique_nodule in enumerate(possible_nodules):
            if idx_unique == 0:
                df_coords_adjusted = pd.DataFrame()
            #if idx_unique==1:break
            df_nodule = df_patient_partX.loc[df_patient_partX['cluster_id']==unique_nodule] # this gives all annotations for this nodule (cluster)

            # FIND THE TRANSFORMED COORDINATES 
            resampling_ratio = [j/i for i,j in zip(np.shape(curr_patient_pixels), np.shape(pix_resampled))]
            #print(f'analyzing id nodule = {unique_nodule}')
            nodule_coords_resamp, success, coord_mask, coord_calc  = nodule_coords_in_small_resampled_versions4_5(df_nodule, 
            resampling_ratio, min_box_x, min_box_y, min_box_channels, slice_middle, ymed_1, ymed_2, 
            box_coords1, box_coords2, nodule_centers_from_image)
            if success:
                df_coords_adj_temp, coord_adj_Z, coord_adj_X, coord_adj_Y, coords_found = nodule_coords_resamp
                df_coords_adjusted = df_coords_adjusted.append(df_coords_adj_temp)

        try:
            df_coords_adjusted['small_coordsZ_resampled'] = df_coords_adjusted.small_coordsZ.values * spacing_orig[0]
        except AttributeError: continue
        df_coords_adjusted.to_csv(f'{path_dest}pylidc_characteristics/{pid}_{block_name}.csv', index=False)
        #df_coords_adjusted_all = df_coords_adjusted_all.append(df_coords_adjusted)
        
        np.save(f'{path_dest}arrays/last/{pid}_{block_name}.npy',last_resampled)
        np.save(f'{path_dest}arrays/orig/{pid}_{block_name}.npy',orig_resampled)
        np.savez_compressed(f'{path_dest}arrays/masks nodules/{pid}_{block_name}',mask_resampled)
        np.savez_compressed(f'{path_dest}arrays/masks lungs/{pid}_{block_name}',mask_lungs_resampled)
        
        
        names_checkup.append(f'{pid}_{block_name}')
        coord_mask_all.append(coord_mask) 
        coord_calc_all.append(coord_calc)

