from typing import Any, Union, Dict, Literal
from pathlib import Path
from numpy.typing import NDArray
from collections import defaultdict
import h5py
import torch
import segmentation_models_pytorch as smp 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
from plotting import FieldPlotter
import matplotlib.colors as mcolors

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
 

class MakePrediction:
        
    def predict(image : NDArray(512)) -> NDArray:
        
        metric = MakePrediction.calculate_metric(image)
                
        IMG_HEIGHT  = 512
        DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
        model_name  = "xception"
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
        return NDWI 
    
    def get_abai(image):     
        B03=MakePrediction.get_band(image,2) 
        B11=MakePrediction.get_band(image,10) 
        B12=MakePrediction.get_band(image,11) 
        
        ABAI = (3*B12 - 2*B11 - 3*B03 ) / ((3*B12 + 2*B11 +3*B03) + 1e-10)
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        cmap = None        
        ax.imshow(ABAI, cmap=cmap)
        plt.show()
        """
        print (B03.mean(),B03.max(),B03.min(),'<< B3')
        print (B11.mean(),B11.max(),B11.min(),'<< B11')
        print (B12.mean(),B12.max(),B12.min(),'<< B12')
         
        print (ABAI.mean(),ABAI.max(),ABAI.min(),'<< ABAI')
        
        return ABAI
    
    def get_water_mask(image):

        NDWI=MakePrediction.get_ndwi(image)     
        water_mask=NDWI
        water_mask[water_mask >= 0] = 1
        water_mask[water_mask < 0] = -1
        return water_mask
    
    def calculate_metric(image):
        # Get the indices 
        ABAI = MakePrediction.get_abai(image)  
        metric = ABAI
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        cmap = None
        ax.imshow(metric, cmap=cmap)
        plt.show()
        """
        # Mask water with min value of metric
        water_mask=MakePrediction.get_water_mask(image) 
        metric[water_mask == 1] = metric.min()
        
        # Normalize & standardize        
        metric_scaled = (2 * norm(metric)) - 1
        metric=metric_scaled 
        
        # Set a threshold 
        #metric[metric <0]=-1
        return metric
    
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
        
 