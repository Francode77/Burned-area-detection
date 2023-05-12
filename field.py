# Class field
import rasterio
from rasterio import plot
from loader import loader
import numpy as np
import os


class DataSource:
   loaded_in_batches = {}

   def get_image(source_file,batch_nr, img_nr, scene_nr):
       batches_for_file=DataSource.loaded_in_batches.setdefault(source_file,{})
       if batch_nr not in batches_for_file:
           DataSource.loaded_in_batches[source_file][batch_nr] = loader(source_file, [batch_nr]) 
       return DataSource.loaded_in_batches[source_file][batch_nr][scene_nr][img_nr,:,:,:]

   def get_band(source_file,batch_nr, img_nr, scene_nr, band_nr):
       batches_for_file=DataSource.loaded_in_batches.setdefault(source_file,{})
       if batch_nr not in batches_for_file:
           DataSource.loaded_in_batches[source_file][batch_nr] = loader(source_file, [batch_nr]) 
       return DataSource.loaded_in_batches[source_file][batch_nr][scene_nr][img_nr,:,:,band_nr]
 
   def get_mask(source_file,batch_nr, img_nr):
       batches_for_file=DataSource.loaded_in_batches.setdefault(source_file,{})
       if batch_nr not in batches_for_file:
           DataSource.loaded_in_batches[source_file][batch_nr] = loader(source_file, [batch_nr]) 
       return DataSource.loaded_in_batches[source_file][batch_nr][2][img_nr,:,:,0]
   
def norm(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min)/(band_max - band_min))

def stand(band):
    band_min, band_max = band.min(), band.max()
    return ((2 * (band - band_min)/(band_max - band_min)) - 1)
    
class Field:
    def __init__(self, source_file, batch_nr, img_nr):     
         
        self.img_nr = img_nr # image in batch
        self.batch_nr = batch_nr  
        self.source_file = source_file
        self.source_name = str(source_file)[5:-4]

    """ RETURN FUNCTIONS """       
 
    def bands(self,scene_nr,band_nr):
        band = DataSource.get_band(self.source_file, self.batch_nr, self.img_nr, scene_nr, band_nr)
        return band
               
    def return_mask(self): 
        mask=DataSource.get_mask(self.source_file, self.batch_nr, self.img_nr) 
        return mask  
    
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
        _,ABAI_after = self.get_bi_abai()  
        metric = ABAI_after
        
        # Mask water with min value of metric
        water_mask=self.get_water_mask() 
        metric[water_mask == 1] = metric.min()
        
        # Normalize & standardize        
        metric_scaled = (2 * norm(metric)) - 1
        metric=metric_scaled 
        
        # Set a threshold 
        #metric[metric <0]=-1
        
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
        with rasterio.open(f'{output_dir}/{self.source_name}_{self.batch_nr}_{self.img_nr}.tif', 'w', **meta) as dst:
        
            # Write the numpy array to the file
            dst.write(metric, 1)
        
        # Get the truth value mask        
        mask_data=self.return_mask()  
        output_dir = os.path.join("data","processed","training_truth") 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Open a new raster file for writing
        with rasterio.open(f'{output_dir}/{self.source_name}_{self.batch_nr}_{self.img_nr}.tif', 'w', **meta) as dst:
        
            # Write the numpy array to the file
            dst.write(mask_data, 1)
