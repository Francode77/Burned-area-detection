"""Field class applies to the two scenes and truth mask in a .hd5f datasource file
"""

# Imports
from classes.datasource import DataSource
import os
import rasterio 
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage.morphology import generate_binary_structure
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Function normalize  
def norm(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min)/(band_max - band_min))
# Function standardize
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

    
    """  INDICES FUNCTIONS
    Functions to get indices from pre- and post fire scenes (0/1) in the dataset
    """
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
        ABAI_after = (3*B12.astype(float) - 2*B11.astype(float) - 3*B03.astype(float) ) / ((3*B12.astype(float) + 2*B11.astype(float) +3*B03.astype(float)) + 1e-10)

        B03=self.bands(1,2) 
        B11=self.bands(1,10) 
        B12=self.bands(1,11) 
        ABAI_before = (3*B12.astype(float) - 2*B11.astype(float) - 3*B03.astype(float) ) / ((3*B12.astype(float) + 2*B11.astype(float) +3*B03.astype(float)) + 1e-10)
          
        return ABAI_before,ABAI_after
    
    
    ### WATER MASK  
    """ Water mask will add both scenes' water regions and return numpy array
        NDWI <=0 will be masked as water
        When vegetation in water causes NDWI > 0 , every pixel surrounded by water is masked
        by using closing method, implemented by dilation and erosion
    """
    def get_water_mask(self):

        NDWI_before,NDWI_after=self.get_ndwi()    
        
        # Check and replace NaN and Inf values with 0
        NDWI_before = np.nan_to_num(NDWI_before, nan=0, posinf=0, neginf=0)
        NDWI_after = np.nan_to_num(NDWI_after, nan=0, posinf=0, neginf=0)

        NDWI_before[NDWI_before <= 0] = -1      
        NDWI_before[NDWI_before > 0] = 1      
        NDWI_after[NDWI_after <= 0] = -1      
        NDWI_after[NDWI_after > 0] = 1
        
        water_mask = NDWI_before + NDWI_after 
        water_mask[water_mask >= 0] = 1
        water_mask[water_mask < 0] = 0
        
       # Exclude regions with vegetation in water
        def update_groups(mask):   
            # Define a structuring element  
            structuring_element = generate_binary_structure(2, 2)
            
            # Perform binary dilation to expand the 0 regions
            dilated = binary_dilation(mask, structure=structuring_element)
            
            # Perform binary erosion to shrink the expanded regions
            eroded = binary_erosion(dilated, structure=structuring_element)
            
            # Set mask = 1 where the eroded matrix is different from the original mask
            updated_mask = np.where(eroded != mask, 1, mask)
            
            return updated_mask         
        
        # Update the groups of 0's surrounded by 1's
        updated_mask = update_groups(water_mask)

        return updated_mask 
    
    
    
    """ METRIC FUNCTIONS """
    """ Calculates the metric and outputs it to a new .tiff file """
    
    def calculate_metric(self):

        # Get the indices 
        _,ABAI_after = self.get_bi_abai()  
        metric = ABAI_after
        
        # Mask water with min value of metric
        water_mask = self.get_water_mask() 
        metric[water_mask == 1] = metric.min()
        
        # Mask fire with min value of metric
        fire_mask = self.active_fire_mask()
        metric[fire_mask == 1] = metric.min()
        
        # Standard deviation        
        metric[metric> 2.23 * np.std(metric)]= 2.23 * np.std(metric)
         
        # Normalize & standardize        
        metric_scaled = (2 * norm(metric)) - 1
        metric = metric_scaled     
        
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

   
    def active_fire_mask(self): 
        
        # Normalized Green Difference Vegetation Index
        B02=self.bands(0,1).astype(float) / 10000
        B03=self.bands(0,2).astype(float) / 10000
        B04=self.bands(0,3).astype(float) / 10000
 
        ## Fire indicator    
        B11=self.bands(0,10).astype(float) / 10000
        B12=self.bands(0,11).astype(float) / 10000
        
        # Structural Analysis of Hydrologic Modeling
        SAHM_INDEX= ((B12.astype(float) - B11.astype(float)) / ((B12.astype(float) + B11.astype(float)))+ 1e-10)
 
        SAHM_mask = (SAHM_INDEX>0.4) | (B12>1)
        B04[SAHM_mask] *= 20
        B03[SAHM_mask] *= 1
        B02[SAHM_mask] *= 1
         
        water_mask = self.get_water_mask()
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

    
    def get_firemask(self):
        
        _,_,_,mask=Field.fire_detection(self)
        return mask
    
    def fire_detection(self):

       # Normalized Green Difference Vegetation Index
       B02=self.bands(0,1).astype(float) / 10000
       B03=self.bands(0,2).astype(float) / 10000
       NGDR = (B02.astype(float) - B03.astype(float))/ (B02.astype(float) + B03.astype(float) + 1e-10)
       
       inverse = (B02.astype(float) - 0.2) / (0.5 - 0.2)
       
       ## Fire indicator    
       B11=self.bands(0,10).astype(float) / 10000
       B12=self.bands(0,11).astype(float) / 10000
       B04=self.bands(0,3).astype(float) / 10000
       
       # Structural Analysis of Hydrologic Modeling
       SAHM_INDEX= ((B12.astype(float) - B11.astype(float)) / (B12.astype(float) + B11.astype(float)) + 1e-10)
 
       INV_mask = (inverse > 1) 
       B04[INV_mask] *= 0.5
       B03[INV_mask] *= 0.5
       B02[INV_mask] *= 20
       
       NGDR_mask = (inverse > 0) & (NGDR > 0) 
       B04[NGDR_mask] = B04.min()
       B03[NGDR_mask] = B03.min()
       B02[NGDR_mask] = B02.max() * 20
   
       SAHM_mask = (SAHM_INDEX>0.4) | (B12>1)
       B04[SAHM_mask] *= 20
       B03[SAHM_mask] *= 1
       B02[SAHM_mask] *= 1
       
       print (SAHM_INDEX.max(),'<<<<<<')
       print (np.count_nonzero(SAHM_INDEX>0.34))
       
       water_mask = self.get_water_mask()
       mask = (water_mask == 1) 
       B04[mask] = B04.min()
       B03[mask] = B03.min()
       B02[mask] = B02.min()
 
       final_mask = ( INV_mask | NGDR_mask | SAHM_mask ) & ~ mask
 
       return B04, B03, B02, final_mask