# Class field

import rasterio
from loader import loader
import matplotlib.pyplot as plt
from rasterio.plot import show
import numpy as np
import os
from skimage.color import rgb2gray

class DataSource:
   loaded_in_batches = {}
   
   def get_image(batch_nr, img_nr, scene_nr):
       if batch_nr not in DataSource.loaded_in_batches:
           DataSource.loaded_in_batches[batch_nr] = loader('./data/train_eval.hdf5', [batch_nr]) 
       return DataSource.loaded_in_batches[batch_nr][scene_nr][img_nr,:,:,:]
   
   def get_band(batch_nr, img_nr, scene_nr, band_nr):
       if batch_nr not in DataSource.loaded_in_batches:
           DataSource.loaded_in_batches[batch_nr] = loader('./data/train_eval.hdf5', [batch_nr]) 
       return DataSource.loaded_in_batches[batch_nr][scene_nr][img_nr,:,:,band_nr]
 
   def get_mask(batch_nr, img_nr):
       if batch_nr not in DataSource.loaded_in_batches:
           DataSource.loaded_in_batches[batch_nr] = loader('./data/train_eval.hdf5', [batch_nr]) 
       return DataSource.loaded_in_batches[batch_nr][2][img_nr,:,:,0]
   
def norm(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min)/(band_max - band_min))

def stand(band):
    band_min, band_max = band.min(), band.max()
    return ((2 * (band - band_min)/(band_max - band_min)) - 1)
    
class Field:
    def __init__(self, batch_nr,img_nr):     
         
        self.img_nr = img_nr # image in batch
        self.batch_nr = batch_nr  
    
    """ PLOT FUNCTIONS """    
    def plot(self,indicator,mask):
        fig, ax = plt.subplots(figsize=(10, 10))  
        cmap=None
        ax.imshow(indicator,cmap=cmap)
        if mask==1:
            mask=self.return_mask()       
            ax.imshow(mask,alpha=.33)
        plt.show()
            
    def bi_plot(self,indicator,before,after,cmap,mask):
    
        fig, axs = plt.subplots(1, 2, figsize=(15, 15))        
 
        axs[0].imshow(after,cmap=cmap)
        axs[0].set_title(f'{indicator} Scene 0 | Fold {self.batch_nr} | Image {self.img_nr}')    
        axs[1].imshow(before,cmap=cmap)
        if mask==1:
            mask=self.return_mask()       
            axs[1].imshow(mask,alpha=.3)
        axs[1].set_title(f'{indicator} Scene 1 | Fold {self.batch_nr} | Image {self.img_nr}')        
        plt.show()    
        
    def bi_plot_mask(self,img): 
        mask=self.return_mask()        
        fig, axs = plt.subplots(1, 2, figsize=(15, 15))        
        cmap=None
        axs[0].imshow(img,cmap=cmap,vmin=-1,vmax=1)
        axs[0].set_title(f'RESULT | Fold {self.batch_nr} | Image {self.img_nr}')        
        axs[1].imshow(mask)
        axs[1].set_title(f'MASK | Fold {self.batch_nr} | Image {self.img_nr}') 
        plt.show()    
            
    def plot_watermask(self):        
        water_mask=self.get_water_mask()
        fig, ax = plt.subplots(figsize=(10, 10)) 
        ax.imshow(water_mask)
         
    def plot_hist(self,metric):
        
        counts, bins = np.histogram(metric.flatten())
        plt.hist(bins[:-1], bins, weights=counts)
        plt.ylim(0,200)
 
    """ PLOT INDICES FUNCTIONS"""
        
    def plot_rgb(self,mask,factor):
        r,g,b=self.get_rgb(0)
        
        # Stack the bands to create an RGB image
        RGB_after = rasterio.plot.reshape_as_image([norm(r), norm(g), norm(b)])
        
        r,g,b=self.get_rgb(1)
        RGB_before= rasterio.plot.reshape_as_image([norm(r), norm(g), norm(b)])
        
        cmap=None
        self.bi_plot("RGB",RGB_before*factor,RGB_after*factor,cmap=cmap,mask=mask)
        
    def plot_abai(self,mask):
         
        B03=self.bands(0,2) 
        B11=self.bands(0,10) 
        B12=self.bands(0,11) 
        ABAI_after = (3*B12 - 2 * B11 - 3 * B03 ) / ((3*B12 + 2*B11 +3*B03) + 1e-10)
        B03=self.bands(1,2) 
        B11=self.bands(1,10) 
        B12=self.bands(1,11) 
        ABAI_before = (3*B12 - 2 * B11 - 3 * B03 ) / ((3*B12 + 2*B11 +3*B03) + 1e-10)
            
        cmap=None
        self.bi_plot("ABAI", ABAI_before, ABAI_after,cmap=cmap,mask=mask)     

         
    """ RETURN FUNCTIONS """       
 
    def bands(self,scene_nr,band_nr):
        band = DataSource.get_band(self.batch_nr, self.img_nr, scene_nr, band_nr)
        return band
               
    def return_mask(self): 
        self.mask=DataSource.get_mask(self.batch_nr, self.img_nr) 
        return self.mask  
    
    def get_rgb(self,scene_nr):    
        r=self.bands(scene_nr,3)
        g=self.bands(scene_nr,2)
        b=self.bands(scene_nr,1)   
        return r,g,b     
    
    def get_water_mask(self):

        NDWI_before,NDWI_after=self.get_ndwi()    
        # Check and replace NaN and Inf values with 0
        NDWI_before = np.nan_to_num(NDWI_before, nan=0, posinf=0, neginf=0)
        NDWI_after = np.nan_to_num(NDWI_after, nan=0, posinf=0, neginf=0)

        NDWI_before[NDWI_before <= 0] = -1      
        NDWI_before[NDWI_before > 0] = 1      
        NDWI_after[NDWI_after <= 0] = -1      
        NDWI_after[NDWI_after > 0] = 1
        
        water_mask=NDWI_before+NDWI_after 
        water_mask[water_mask >= 0] = 1
        water_mask[water_mask < 0] = -1
        return water_mask
    
    def get_ndwi(self):        
        B03=self.bands(0,2)
        B08=self.bands(0,7)         
        NDWI_after = (B03.astype(float) - B08.astype(float)) / (B03.astype(float) + B08.astype(float) + 1e-10)   
        B03=self.bands(1,2)
        B08=self.bands(1,7) 
        NDWI_before= (B03.astype(float) - B08.astype(float)) / (B03.astype(float) + B08.astype(float) + 1e-10)
        return NDWI_before,NDWI_after

    def get_bi_abai(self):        
        B03=self.bands(0,2) 
        B11=self.bands(0,10) 
        B12=self.bands(0,11) 
        ABAI_after = (3*B12 - 2 * B11 - 3 * B03 ) / ((3*B12 + 2*B11 +3*B03) + 1e-10)
        B03=self.bands(1,2) 
        B11=self.bands(1,10) 
        B12=self.bands(1,11) 
        ABAI_before = (3*B12 - 2 * B11 - 3 * B03 ) / ((3*B12 + 2*B11 +3*B03) + 1e-10)
           
        return ABAI_before,ABAI_after
    
    """ METRIC FUNCTIONS """
    
    def calculate_metric(self):
        # Get the indices 
        ABAI_before,ABAI_after = self.get_bi_abai()  
        metric = ABAI_after
        
        self.plot_hist(metric)
 
        # Mask water with min value of metric
        water_mask=self.get_water_mask() 
        metric[water_mask == 1] = metric.min()
        
        self.plot_hist(metric)
        
        # Normalize & standardize        
        metric_scaled = (2 * norm(metric)) - 1
        metric=metric_scaled 
        count_ones = np.count_nonzero(metric > 0.5)

        print("Number of ones:", count_ones)
        
        
        self.plot_hist(metric)
          
        
        # Set a threshold 
        metric[metric <0]=-1
        
        return metric
        
    def write_metric(self):
        
        metric=self.calculate_metric()
        
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
        output_dir = os.path.join("data","processed","training_scene") 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Open a new raster file for writing
        with rasterio.open(f'{output_dir}/{self.batch_nr}_{self.img_nr}.tif', 'w', **meta) as dst:
        
            # Write the numpy array to the file
            dst.write(metric, 1)
        
        # Get the truth value mask        
        mask_data=self.return_mask()  
        output_dir = os.path.join("data","processed","training_truth") 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Open a new raster file for writing
        with rasterio.open(f'{output_dir}/{self.batch_nr}_{self.img_nr}.tif', 'w', **meta) as dst:
        
            # Write the numpy array to the file
            dst.write(mask_data, 1)
        
    def plot_metric(self,mask):
 
        metric=self.calculate_metric()
        
        # Plot the output
        self.plot(metric,mask=0) 
        
        # Plot output with mask comparison
        self.bi_plot_mask(metric)
        
        
        
 

        
     