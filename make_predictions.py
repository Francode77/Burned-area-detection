from typing import Any, Union, Dict, Literal
from numpy.typing import NDArray
import torch
import segmentation_models_pytorch as smp 
import torchvision.transforms as transforms 
import os
import numpy as np
import rasterio
import matplotlib.colors as mcolors

from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage.morphology import generate_binary_structure

# Define custom colors
colors = ['purple', 'yellow']

# Create a custom colormap
cmap = mcolors.ListedColormap(colors)
        
def norm(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min)/(band_max - band_min))

def stand(band):
    band_min, band_max = band.min(), band.max()
    return ((2 * (band - band_min)/(band_max - band_min)) - 1)

# Remove groups of 0's surrounded by 1's
def update_groups(matrix):   
    # Define the structuring element (larger size)
    structuring_element = generate_binary_structure(2, 2)
    # Perform binary dilation to expand the 0 regions
    dilated = binary_dilation(matrix , structure=structuring_element)
    
    # Perform binary erosion to shrink the expanded regions
    eroded = binary_erosion(dilated,  structure=structuring_element)
    
    # Find the locations where the eroded matrix is different from the original matrix
    updated_mask = np.where(eroded != matrix, 1, matrix)
    
    return updated_mask         


class MakePrediction:
        
    def predict(image : NDArray(512), model_name : str) -> NDArray:
        
        metric = MakePrediction.calculate_metric(image)
                
        IMG_HEIGHT  = 512
        DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
        model_name  = model_name
        MODEL_PATH  = f"{model_name}_trained_{IMG_HEIGHT}px.pt"
        
        model = smp.Unet(encoder_name=model_name, in_channels=1, classes=1, activation=None).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval() 
        transform_ = transforms.Compose([transforms.ToTensor(), 
            transforms.Resize((IMG_HEIGHT,IMG_HEIGHT),antialias=True)])
        # get normalized image
        img_tensor = transform_(metric).float()
        img_tensor = img_tensor.unsqueeze_(0)
        img_tensor = img_tensor.to(DEVICE)
 
        result = ((torch.sigmoid(model(img_tensor.to(DEVICE)))) >0.5).float()
         
        mask = result[0].cpu().numpy()
        mask = mask[0]  
        bool_output = mask.astype(bool)
  
        return metric, mask, bool_output
    
    def get_band(image, band_nr : int):
        band=image[:,:,band_nr]
        return band

    def get_ndwi(image):        
        
        B03=MakePrediction.get_band(image,2)
        B08=MakePrediction.get_band(image,7)         
        NDWI = (B03.astype(float) - B08.astype(float)) / (B03.astype(float) + B08.astype(float) + 1e-10) 
        # Check and replace NaN and Inf values with 0
        NDWI = np.nan_to_num(NDWI, nan=0, posinf=0, neginf=0)  
        return NDWI 
    
    def get_abai(image):     
        B03=MakePrediction.get_band(image,2) 
        B11=MakePrediction.get_band(image,10) 
        B12=MakePrediction.get_band(image,11)         
        ABAI = (3*B12.astype(float) - 2*B11.astype(float) - 3*B03.astype(float) ) / ((3*B12.astype(float) + 2*B11.astype(float) +3*B03.astype(float)) + 1e-10)
        return ABAI
    
    def get_water_mask(image):
        NDWI=MakePrediction.get_ndwi(image)     
        water_mask=NDWI
        water_mask[water_mask >= 0] = 1
        water_mask[water_mask < 0] =0   

        # Update groups of 0's surrounded by 1's
        water_mask = update_groups(water_mask)
        
        return water_mask     
    
    def active_fire_mask(image): 
        
        # Normalized Green Difference Vegetation Index
        B02=MakePrediction.get_band(image,1).astype(float) / 10000
        B03=MakePrediction.get_band(image,2).astype(float) / 10000
        B04=MakePrediction.get_band(image,3).astype(float) / 10000
 
        ## Fire indicator    
        B11=MakePrediction.get_band(image,10).astype(float) / 10000
        B12=MakePrediction.get_band(image,11).astype(float) / 10000
        
        # Structural Analysis of Hydrologic Modeling
        SAHM_INDEX= ((B12.astype(float) - B11.astype(float)) / ((B12.astype(float) + B11.astype(float)))+ 1e-10)
 
        SAHM_mask = (SAHM_INDEX>0.4) | (B12>1)
        B04[SAHM_mask] *= 20
        B03[SAHM_mask] *= 1
        B02[SAHM_mask] *= 1
         
        water_mask = MakePrediction.get_water_mask(image) 
        mask = (water_mask == 1) 
        B04[mask] = B04.min()
        B03[mask] = B03.min()
        B02[mask] = B02.min()
 
        fire_mask = SAHM_mask
        
        # Dilate the region 
        structuring_element = generate_binary_structure(2, 2)
        # Perform binary dilation to expand the 0 regions
        dilated = binary_dilation(fire_mask , iterations=6, structure=structuring_element)
        fire_mask = dilated
        
        
        return fire_mask
    
    def calculate_metric(image):
        # Get the indices 
        ABAI = MakePrediction.get_abai(image)  
        metric = ABAI
        
        # Water surface detection
        water_mask=MakePrediction.get_water_mask(image) 
        metric[water_mask == 1] = metric.min()
 
        # Fire detection
        fire_mask = MakePrediction.active_fire_mask(image)
        metric[fire_mask == 1] = metric.min()
        
        # Standard deviation        
        metric[metric> 2.23 * np.std(metric)]= 2.23 * np.std(metric)
         
        # Normalize & standardize        
        metric_scaled = (2 * norm(metric)) - 1
        metric=metric_scaled 
     
        # Set a threshold 
        #metric[metric <0]=-1
        
        return metric
    
    # Not necessary anymore
    def write_eval_metric(image,uuid):
        
        metric=MakePrediction.calculate_metric(image)
        
        """SAVE AS TIFF GEOTIFF"""
        
        # Define some metadata for the output file
        meta = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': -1,
            'width': 512,
            'height': 512,
            'count': 1,
            'crs': 'EPSG:4326',
            'transform': [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        }
        output_dir = os.path.join("data","evaluation") 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Open a new raster file for writing
        with rasterio.open(f'{output_dir}/{uuid}.tif', 'w', **meta) as dst:
        
            # Write the numpy array to the file
            dst.write(metric, 1)
        
 