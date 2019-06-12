import numpy as np 
import scipy
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.morphology import erosion, dilation, binary_opening, ball

#%%
# load in a scan, transform to HU, and output the accurate spacing 
def custom_load_scan_to_HU(scan):
    
    # first get the list of dicoms 
    sliceList = scan.load_all_dicom_images()
    
    # then get the volume 
    vol_orig = np.stack([s.pixel_array for s in sliceList]) # stack all the slices into one ndarray
    
    # calculate the spacing of this volume, using spacing, not thickness!
    spacing = np.array([float(scan.slice_spacing), float(sliceList[0].PixelSpacing[0]), float(sliceList[0].PixelSpacing[1])], dtype=np.float32)
    
    # convert the volume to HU based on slope and intercept in the dicom
    vol_HU = vol_orig.astype(np.int16)
    
    # Convert to Hounsfield units (HU) according to the dicom information
    for slice_number in range(len(sliceList)):
        
        intercept = sliceList[slice_number].RescaleIntercept # dicom info has conversion factors
        slope = sliceList[slice_number].RescaleSlope # dicom info has conversion factors
        
        if slope != 1: # apply the intercept, if non-one
            vol_HU[slice_number] = slope * vol_HU[slice_number].astype(np.float64)
            vol_HU[slice_number] = vol_HU[slice_number].astype(np.int16) # remember to cast back to int16 after multiplying by a float
            
        vol_HU[slice_number] += np.int16(intercept) # apply the intercept
        
    # set the values outside the FOV to be air (-1000)
    outside_fov = vol_HU[0,0,0]
    vol_HU[vol_HU == outside_fov] = -1000
    
    return vol_HU, spacing

#%%

# resize the image volume to a certain size (1 x 1 x 1 mm3) using zoom method
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    # modified by OMM - floats added
    spacing = np.array([float(scan[0].SliceThickness), float(scan[0].PixelSpacing[0]), float(scan[0].PixelSpacing[1])], dtype=np.float32)
    
    resize_factor = spacing / np.asarray(new_spacing)
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape) # rounding here, so don't get exact [1,1,1] spacing
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    imageOut = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return imageOut, new_spacing    

# resample image to 1x1x1 using grid method
def resample_grid(image, spacing, new_spacing=[1,1,1],method='linear'):
        
    x, y, z = [spacing[k] * np.arange(image.shape[k]) for k in range(3)]  # original grid in mm
    f = scipy.interpolate.RegularGridInterpolator((x, y, z), image, method=method)    # interpolator    
#    print('Interpolating')

    dx, dy, dz = new_spacing    # new step sizes
    new_grid = np.mgrid[0:x[-1]:dx, 0:y[-1]:dy, 0:z[-1]:dz]   # new grid
    new_grid = np.moveaxis(new_grid, (0, 1, 2, 3), (3, 0, 1, 2))  # reorder axes for evaluation
    imageOut = f(new_grid)
    # convert back to the same type as input (if it was an int, round first!)
    dataType = image.dtype
    if np.issubdtype(image[0,0,0],np.signedinteger) or np.issubdtype(image[0,0,0],np.unsignedinteger):
        imageOut = np.round(imageOut)
        
    imageOut = imageOut.astype(dataType)
    
    return imageOut, new_spacing

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

#%%
# largest_label_volume 
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True) # gets the unique values and the number of occurences of each (this is used on labelled images, cf bwconncomp)

    counts = counts[vals != bg] # get the number of occurences for non-background regions
    vals = vals[vals != bg] # only keep these values too

    if len(counts) > 0:
        return vals[np.argmax(counts)] # return the value of vals with the highest count
    else:
        return None    

# function to apply a z-only morphological closing (dilation followed by erosion)
def imcloseZ(binary_image_orig,selemZ_width):
    
    # if we have a width of zero, return the original image
    if selemZ_width == 0:
        binary_image = binary_image_orig.copy()
        return binary_image
    elif selemZ_width > 0:
        selem = np.ones((selemZ_width,1,1),dtype='int8')
        binary_image = erosion(dilation(binary_image_orig,selem),selem)
        return binary_image

# function to remove the bed (or rather keep only the body) in an image
def removeBed(image):
    # primary assumption: that the centre voxel in the image is within the 
    # patient's body
    
    # 1) binarize the 3D image
    otsuThresh = threshold_otsu(image)
    binary_image = np.array(image > otsuThresh, dtype=np.int8)
    
    # 1.5) perform a small morphological opening for the cases where the bed 
    # and the patient are connected
    binary_image = binary_opening(binary_image,ball(3))
    
    # 2) fill the holes on each 2D slice (fill the lungs)
    binary_image_2D_hole_filled = np.zeros_like(binary_image)
    for ind, binary_slice in enumerate(binary_image):
        binary_image_2D_hole_filled[ind] = scipy.ndimage.morphology.binary_fill_holes(binary_image[ind])
        
    # 3) label the image and find the label of the central voxel (assumed to be
    # within the body)
    labels_filled = measure.label(binary_image_2D_hole_filled,neighbors=8)
    midPoint = tuple([int(np.round(labels_filled.shape[i]/2.0)) for i in range(0,3)])
    label_body = labels_filled[midPoint]
    
    # 4) get the mask of this label, assuming this corresponds to the body
    bodyMask = np.where(labels_filled != label_body)
    
    # 5) apply the mask to remove everything outside the body (setting to -1000
    # for now)
    imageOut = image.copy()
    imageOut[bodyMask] = -1000 # replace with air
    
    return imageOut

# the full segmentation function
def segment_lung_mask(image, fill_lung_structures=True):
    
    image = removeBed(image) # function to remove the bed from the image
    
    numSegVoxels = 0
    selemZ_width = -1
        
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    # use otsu method
    otsuThresh = threshold_otsu(image)
    binary_image_orig = np.array(image > otsuThresh, dtype=np.int8)+1 #this is a mask of everything that is soft tissue (1 or 2 with the +1)

    # as long as the number of segmented voxels is less than 1e6, increment the morphological closing element and try again
    while numSegVoxels < 1e6 :
        selemZ_width += 1
        
        if selemZ_width > 0:
            print(f'Require a morphological closing of size {selemZ_width}')
        
        # apply a morphological closing if necessary
        binary_image = imcloseZ(binary_image_orig,selemZ_width)

        # label the regions
        # (pad image with zeros though, since we a priori know that the outside
        # should all be connected)
        # (though we do not pad in the axial direction, since we risk connecting
        # trachea to exterior)
        labels = measure.label(np.pad(binary_image,((0,0),(1,1),(1,1)),'constant',constant_values=1),neighbors=8)[0:,1:-1,1:-1] # label function in skimage does a connected-component labelling of the binary image 
       
        
        # Pick the pixel in every corner to determine which label is air.
        #   Improvement: take the labels from all 8 corners to catch when the patient fills the fov
        background_label = []
        background_label.append(labels[0,0,0])
        background_label.append(labels[0,0,-1])
        background_label.append(labels[0,-1,0])
        background_label.append(labels[-1,0,0])
        background_label.append(labels[-1,0,-1])
        background_label.append(labels[-1,-1,0])
        background_label.append(labels[0,-1,-1])
        background_label.append(labels[-1,-1,-1])
        
        background_label = np.unique(background_label)
        
        #Fill the air around the person with a value of 2 (which is the same as most of the body)
        for i in range(0,len(background_label)):
            binary_image[labels == background_label[i]] = 2 
        
        # Method of filling the lung structures (that is superior to something like 
        # morphological closing)
        if fill_lung_structures:
            # For every slice we determine the largest solid structure
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice - 1
                labeling = measure.label(axial_slice)
                l_max = largest_label_volume(labeling, bg=0) # l_max is the label of the largest value, that is the non-lung area
                
                if l_max is not None: #This slice contains some lung
                    binary_image[i][labeling != l_max] = 1
    
        
        binary_image -= 1 #Make the image actual binary
        binary_image = 1-binary_image # Invert it, lungs are now 1
        
        # Remove other air pockets insided body by removing small regions 
        # SE edit: compare the largest and second largest in case the two lungs are not connected
        # due to the morphological closing
        labels = measure.label(binary_image, background=0)
        
        regionProps = measure.regionprops(labels)
        regionAreas = [r.area for r in regionProps]
        
        # sort the areas to get the two largest ones
        sortedAreas,sortedInd = np.sort(regionAreas)[::-1][0:2],np.argsort(regionAreas)[::-1][0:2]
        
        # if the two areas of of similar size we assume they are both lung
        # otherwise if the larger area is greater than 2x larger than the next
        # largest, assume that both lungs are in one area
        if (sortedAreas[0]/sortedAreas[1] > 2):
            binary_image[labels != (sortedInd[0] + 1)] = 0
        else: 
            binary_image[(labels != (sortedInd[0] + 1)) & (labels != (sortedInd[1] + 1))] = 0
            
        numSegVoxels = np.count_nonzero(binary_image)
 
    return binary_image, selemZ_width
