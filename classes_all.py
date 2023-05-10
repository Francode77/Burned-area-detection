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
        
 
    def bands(self,scene_nr,band_nr):
        band = DataSource.get_band(self.batch_nr, self.img_nr, scene_nr, band_nr)
        return band
    
    def get_rgb(self,scene_nr):    
        r=self.bands(scene_nr,3)
        g=self.bands(scene_nr,2)
        b=self.bands(scene_nr,1)   
        return r,g,b     
           
    def return_mask(self): 
        self.mask=DataSource.get_mask(self.batch_nr, self.img_nr) 
        return self.mask   
    
    def plot(self,indicator,mask):
        fig, ax = plt.subplots(figsize=(10, 10))  
        cmap=None
        ax.imshow(indicator,cmap=cmap)
        if mask==1:
            mask=self.return_mask()       
            ax.imshow(mask,alpha=.33)
        plt.show()
        
    def plot_watermask(self):
        
        water_mask=self.get_water_mask()
        fig, ax = plt.subplots(figsize=(10, 10)) 
        ax.imshow(water_mask)
        
            
    def bi_plot_mask(self,img): 
        
        mask=self.return_mask()       
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 15))        
        cmap=None
        axs[0].imshow(img,cmap=cmap,vmin=-1,vmax=1)
        axs[0].set_title(f'RESULT | Fold {self.batch_nr} | Image {self.img_nr}')        
        axs[1].imshow(mask)
        axs[1].set_title(f'MASK | Fold {self.batch_nr} | Image {self.img_nr}') 
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
    """ PLOT INDICES FUNCTIONS"""
        
    def plot_rgb(self,mask,factor):
        r,g,b=self.get_rgb(0)
        
        # Stack the bands to create an RGB image
        RGB_after = rasterio.plot.reshape_as_image([norm(r), norm(g), norm(b)])
        
        r,g,b=self.get_rgb(1)
        RGB_before= rasterio.plot.reshape_as_image([norm(r), norm(g), norm(b)])
        
        cmap=None
        self.bi_plot("RGB",RGB_before*factor,RGB_after*factor,cmap=cmap,mask=mask)
        
    def plot_ndvi(self,mask):
        r_before=self.bands(1,3)
        nir_before=self.bands(1,7)
        NDVI_before = (nir_before - r_before) / (nir_before + r_before) 
   
        r_after=self.bands(0,3)
        nir_after=self.bands(0,7)
        NDVI_after = (nir_after - r_after) / (nir_after + r_after) 
        
        cmap='RdYlGn'
        self.bi_plot("NDVI",NDVI_before,NDVI_after,cmap=cmap,mask=mask)
    def plot_gndvi(self,mask):
        
        B03=self.bands(1,2)
        B08=self.bands(1,7)
        GNDVI_before = (B08 - B03) / (B08 + B03)
        B03=self.bands(0,2)
        B08=self.bands(0,7)
        GNDVI_after = (B08 - B03) / (B08 + B03)
        
        cmap='RdYlGn'
        self.bi_plot("GNDVI",GNDVI_before,GNDVI_after,cmap=cmap,mask=mask)
        
    def plot_bsi(self,mask):
        B02=self.bands(0,1)
        B04=self.bands(0,3)
        B08=self.bands(0,7)
        B11=self.bands(0,11)
        BSI_after = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))
 
        B02=self.bands(1,1)
        B04=self.bands(1,3)
        B08=self.bands(1,7)
        B11=self.bands(1,11)
        BSI_before = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))

        cmap='RdYlGn'
        self.bi_plot("BSI",BSI_before,BSI_after,cmap=cmap,mask=mask)
        
    def plot_mi(self,mask):   
        B8A=self.bands(1,8)
        B11=self.bands(1,10)
        MI_before=(B8A-B11)/(B8A+B11)
        B8A=self.bands(0,8)
        B11=self.bands(0,10)
        MI_after=(B8A-B11)/(B8A+B11)
 
        cmap='RdYlGn'
        self.bi_plot("MI",MI_before,MI_after,cmap=cmap,mask=mask)
        
    def plot_avi(self,mask):
        
        B04=self.bands(1,3)
        B08=self.bands(1,7)
        AVI_before=(B08 * (1 - B04)*(B08 - B04))**1/3
        B04=self.bands(0,3)
        B08=self.bands(0,7)
        AVI_after=(B08 * (1 - B04)*(B08 - B04))**1/3
        
        cmap='RdYlGn'
        self.bi_plot("AVI",AVI_before,AVI_after,cmap=cmap,mask=mask)
        
    def plot_savi(self,mask):
        B04=self.bands(1,3)
        B08=self.bands(1,7)
        SAVI_before=(B08 - B04) / (B08 + B04 + 0.428) * (1.428)
        B04=self.bands(0,3)
        B08=self.bands(0,7)
        SAVI_after=(B08 - B04) / (B08 + B04 + 0.428) * (1.428)
        cmap='RdYlGn'
        self.bi_plot("SAVI",SAVI_before,SAVI_after,cmap=cmap,mask=mask)
        
    def plot_ndmi(self,mask):
        
        B08=self.bands(0,7)
        B11=self.bands(0,11)
        NDMI_after=(B08 - B11) / (B08 + B11)
        
        B08=self.bands(1,7)
        B11=self.bands(1,11)
        NDMI_before=(B08 - B11) / (B08 + B11)
        cmap='RdYlGn'
        self.bi_plot("NDMI",NDMI_before,NDMI_after,cmap=cmap,mask=mask)
        
    def plot_gci(self,mask):
        B03=self.bands(0,2)
        B09=self.bands(0,9)
        GCI_after= (B09 / B03) -1
        B03=self.bands(1,2)
        B09=self.bands(1,9)
        GCI_before= (B09 / B03) -1
        cmap='RdYlGn'
        self.bi_plot("GCI",GCI_before,GCI_after,cmap=cmap,mask=mask)
             
    def plot_nbri(self,mask):
        B08=self.bands(0,7)
        B12=self.bands(0,11)
        NBRI_after=-1*(B08 - B12) / (B08 + B12)
        B08=self.bands(1,7)
        B12=self.bands(1,11)
        NBRI_before=-1*((B08 - B12) / (B08 + B12))
        cmap=None
        self.bi_plot("NBRI",NBRI_before,NBRI_after,cmap=cmap,mask=mask)
          
    def plot_ndwi(self,mask):
        
        B03=self.bands(0,2)
        B08=self.bands(0,7)
        
        NDWI_after=(B03 - B08)/ (B03 + B08)
        B03=self.bands(1,2)
        B08=self.bands(1,7)
        NDWI_before=(B03 - B08)/ (B03 + B08)
        
        cmap=None
        self.bi_plot("NDWI",NDWI_before,NDWI_after,cmap=cmap,mask=mask)
 
    def plot_bai(self,mask):
     
        RBR= 0.45
        NIRBR= 0.1
        RED=self.bands(0,3) /10000 # Red 
        NIR=self.bands(0,7) /10000 # NIR
        #BAI_after = 1 / ((RED - RBR) ** 2 + (NIR - NIRBR) ** 2)    
        BAI_after=1/((0.1 -RED)**2 + (0.06 - NIR)**2)
        #BAI_after = 1 / (((RED.astype(float) - RBR) ** 2 + (NIR.astype(float) - NIRBR) ** 2) + 1e-10)             
        RED=self.bands(1,3) /10000 # Red
        NIR=self.bands(1,7) /10000 # NIR
        #BAI_before = 1 / ((RED - RBR) ** 2 + (NIR - NIRBR) ** 2)  
        BAI_before=1/((0.1 -RED)**2 + (0.06 - NIR)**2) 
        #BAI_before = 1 / (((RED.astype(float) - RBR) ** 2 + (NIR.astype(float) - NIRBR) ** 2) + 1e-10)     
        print (RED.mean(),BAI_after.mean())
        cmap=None
        self.bi_plot("BAI", BAI_before, norm(BAI_after),cmap=cmap,mask=mask)
        
        
    def plot_abai(self,mask):
         
        B03=self.bands(0,2) 
        B11=self.bands(0,10) 
        B12=self.bands(0,11) 
        ABAI_after = (3*B12 - 2 * B11 - 3 * B03 ) / (3*B12 + 2*B11 +3*B03)
        B03=self.bands(1,2) 
        B11=self.bands(1,10) 
        B12=self.bands(1,11) 
        ABAI_before = (3*B12 - 2 * B11 - 3 * B03 ) / (3*B12 + 2*B11 +3*B03)
           
        cmap=None
        self.bi_plot("ABAI", ABAI_before, ABAI_after,cmap=cmap,mask=mask)
        
 
    def plot_NBRI_delta(self):
        B08=self.bands(0,7)
        B12=self.bands(0,11)
        NBRI_after=(B08 - B12) / (B08 + B12)
        B08=self.bands(1,7)
        B12=self.bands(1,11)
        NBRI_before=(B08 - B12) / (B08 + B12)
        
        NBRI_delta=NBRI_before-NBRI_after
        
        mean_delta=NBRI_after.mean()-NBRI_before.mean()
        NBRI_after_norm=NBRI_after-mean_delta
        NBRI_before_norm=NBRI_before
             
        NBRI_delta_norm=NBRI_before_norm-NBRI_after_norm
            
        print (mean_delta)
        fig, ax = plt.subplots(figsize=(10, 10))  
        cmap=None
        ax.imshow(NBRI_delta,cmap=cmap)
        fig, ax = plt.subplots(figsize=(10, 10))  
        cmap='RdYlGn'
        ax.imshow(NBRI_delta_norm,cmap=cmap)

    def plot_delta_rgb(self,mask):
        RGB_before,RGB_after=self.get_bi_rgb(mask=mask)     
        mean_delta=RGB_after.mean()- RGB_before.mean()
        RGB_before_norm=RGB_before-mean_delta
        delta_RGB=RGB_before_norm-RGB_after
        print (mean_delta)
        self.plot(delta_RGB*22,mask=mask)
    
    def get_delta_gray(self):
        """ Return delta GRAY before - after"""
        RGB_before,RGB_after=self.get_bi_rgb()
        GRAY_before=rgb2gray(RGB_before)
        GRAY_after=rgb2gray(RGB_after)
        mean_delta=GRAY_after.mean()- GRAY_before.mean()
        GRAY_before_norm=GRAY_before-mean_delta
        delta_GRAY=GRAY_before_norm-GRAY_after
        return delta_GRAY
        
        
    """ RETURN FUNCTIONS """      
    def get_bi_rgb(self):
        r,g,b=self.get_rgb(0)        
        RGB_after = rasterio.plot.reshape_as_image([norm(r), norm(g), norm(b)])   
        r,g,b=self.get_rgb(1)
        RGB_before= rasterio.plot.reshape_as_image([norm(r), norm(g), norm(b)])  
        return RGB_before,RGB_after
    
    def get_ndwi(self):        
        B03=self.bands(0,2)
        B08=self.bands(0,7)        
        #NDWI_after=(B03 - B08)/ (B03 + B08)
        NDWI_after = (B03.astype(float) - B08.astype(float)) / (B03.astype(float) + B08.astype(float) + 1e-10)
        
        B03=self.bands(1,2)
        B08=self.bands(1,7)
        #NDWI_before=(B03 - B08)/ (B03 + B08)
        NDWI_before= (B03.astype(float) - B08.astype(float)) / (B03.astype(float) + B08.astype(float) + 1e-10)
        
        return NDWI_before,NDWI_after
        
    def get_bi_bsi(self):
        B02=self.bands(0,1)
        B04=self.bands(0,3)
        B08=self.bands(0,7)
        B11=self.bands(0,11)
        #BSI_after = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))
        BSI_after = ((B11.astype(float) + B04.astype(float)) - (B08.astype(float) + B02.astype(float))) / ((B11.astype(float) + B04.astype(float)) + (B08.astype(float) + B02.astype(float)) + 1e-10)

        B02=self.bands(1,1)
        B04=self.bands(1,3)
        B08=self.bands(1,7)
        B11=self.bands(1,11)
        #BSI_before = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))
        BSI_before = ((B11.astype(float) + B04.astype(float)) - (B08.astype(float) + B02.astype(float))) / ((B11.astype(float) + B04.astype(float)) + (B08.astype(float) + B02.astype(float)) + 1e-10)

        return BSI_before,BSI_after

    def get_bi_nbri(self):
        B08=self.bands(0,7)
        B12=self.bands(0,11)
        NBRI_after = (B08.astype(float) - B12.astype(float)) / (B08.astype(float) + B12.astype(float) + 1e-10)
        #NBRI_after=(B08 - B12) / (B08 + B12)
        B08=self.bands(1,7)
        B12=self.bands(1,11)
        NBRI_before = (B08.astype(float) - B12.astype(float)) / (B08.astype(float) + B12.astype(float) + 1e-10)
        #NBRI_before=(B08 - B12) / (B08 + B12) 
 
        return NBRI_before,NBRI_after

    def get_bi_ndmi(self):
        
        B08=self.bands(0,7)
        B11=self.bands(0,10)
        #NDMI_after=(B08 - B11) / (B08 + B11)
        NDMI_after = (B08.astype(float) - B11.astype(float)) / (B08.astype(float) + B11.astype(float) + 1e-10)
        
        B08=self.bands(1,7)
        B11=self.bands(1,10)
        #NDMI_before=(B08 - B11) / (B08 + B11)
        NDMI_before = (B08.astype(float) - B11.astype(float)) / (B08.astype(float) + B11.astype(float) + 1e-10)
        return NDMI_before,NDMI_after
        
    def get_bi_gndvi(self):
        
        B03=self.bands(1,2)
        B08=self.bands(1,7)
        #GNDVI_before = (B08 - B03) / (B08 + B03)
        GNDVI_before = (B08.astype(float) - B03.astype(float)) / (B08.astype(float) + B03.astype(float) + 1e-10)
        B03=self.bands(0,2)
        B08=self.bands(0,7)
        #GNDVI_after = (B08 - B03) / (B08 + B03)
        GNDVI_after = (B08.astype(float) - B03.astype(float)) / (B08.astype(float) + B03.astype(float) + 1e-10)
        return GNDVI_before,GNDVI_after 
    
     
    def get_bi_bai(self):
     
        RBR= 0.45
        NIRBR= 0.1
        RED=self.bands(0,3) # Red
        NIR=self.bands(0,7) # NIR
        BAI_after  = 1 / (((RED.astype(float) - RBR) ** 2 + (NIR.astype(float) - NIRBR) ** 2) + 1e-10)   
        BAI_after=1/((0.1 -RED)**2 + (0.06 - NIR)**2)
        RED=self.bands(1,3) # Red
        NIR=self.bands(1,7) # NIR
        BAI_before = 1 / (((RED.astype(float) - RBR) ** 2 + (NIR.astype(float) - NIRBR) ** 2) + 1e-10)  
        BAI_before=1/((0.1 -RED)**2 + (0.06 - NIR)**2)
        return BAI_before, BAI_after
         
    
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

    def get_delta_positive(self,after,before):
        mean_delta=after.mean()- before.mean()
        
        before_norm=before-mean_delta
        delta=before_norm-after
        water_mask=self.get_water_mask()
        delta[water_mask == 1] = -1
        return delta
    
    def get_delta_negative(self,after,before):
        mean_delta=before.mean()- after.mean()
        after_norm=after-mean_delta
        delta=after_norm-before   
        water_mask=self.get_water_mask()
        delta[water_mask == 1] = -1
        return delta
    
    """ METRIC FUNCTIONS """
    
    def calculate_metric(self):
       """         
       Fire regions are darker (negative) or brighter (positive) in indices
       
       1) get the difference of before and after (delta negative/positive)
       2) sum these indicators      
       3) mask the water   
       4) Set a threshold on the result so all values below threshold are -1
   
       """
       # Get the indices
       NBRI_before,NBRI_after = self.get_bi_nbri() 
       BSI_before,BSI_after = self.get_bi_bsi()
       NDMI_before,NDMI_after = self.get_bi_ndmi()
       GNDVI_before,GNDVI_after = self.get_bi_gndvi()
       BAI_before,BAI_after = self.get_bi_bai() 
       ABAI_before,ABAI_after = self.get_bi_abai() 
       # Get the total number of elements
       flat_arr=NBRI_before.flatten()
       size = flat_arr.size 
       
       # Get the number of non-zero elements
       non_zero = np.count_nonzero(flat_arr)
       
       # Calculate the percentage of non-zero elements
       percentage = non_zero / size * 100
       
       print (percentage)
       if percentage >90:
           # Calculate the after/before delta for each indicator
           # Fire region has positive values in these indicators        
           NBRI_delta=self.get_delta_positive(stand(NBRI_after), stand(NBRI_before))
           BAI_delta=self.get_delta_negative(stand(BAI_after),stand(BAI_before))
           ABAI_delta=self.get_delta_negative(stand(ABAI_after),stand(ABAI_before))

           # Fire has negative values in these indicators
           BSI_delta=self.get_delta_negative(stand(BSI_after), stand(BSI_before))
           NDMI_delta=self.get_delta_positive(stand(NDMI_after), stand(NDMI_before))
           GNDVI_delta=self.get_delta_positive(stand(GNDVI_after),stand(GNDVI_before))
           
           GRAY_delta=self.get_delta_gray()
           
           # Sum positive deltas and inverse of negative deltas
           delta= 2* stand(BSI_delta) + 5* stand(NBRI_delta) - stand(NDMI_delta) - 2 * stand(GNDVI_delta)  - 2 * stand (BAI_delta) + 2 * stand(BAI_after) + 0 * stand(NBRI_after)
           
           
           metric = 3* stand(ABAI_delta) + 5* stand (ABAI_after) + stand (GNDVI_delta) + stand(NBRI_delta) - 2 * stand (NDMI_delta)  
           #delta=BSI_delta + NBRI_delta - NDMI_delta - GNDVI_delta
           #delta= stand(BSI_delta) + 1* stand(NBRI_delta) - stand(NDMI_delta) - 1* stand(GNDVI_delta) +( 0* stand (BAI_delta)) +( 0* stand(GRAY_delta))
           #delta=BSI_delta + NBRI_delta - NDMI_delta - GNDVI_delta
           #delta = stand(ABAI_after)
       else:
           metric=0
           return metric
       
       #self.plot(BAI_delta,mask=0) 
       """
       self.plot(BSI_delta,mask=0)         
       self.plot(NBRI_delta,mask=0)        
       self.plot(NDMI_delta,mask=0) 
       self.plot(GNDVI_delta,mask=0) 
       self.plot(BAI_delta,mask=0) 
       self.plot(stand(ABAI_after),mask=0) 
       self.plot(ABAI_delta,mask=0) 
       """

       # Mask water
       water_mask=self.get_water_mask() 
       metric[water_mask == 1] = metric.min()
       
       # Normalize
       metric_norm=(metric-metric.min())/(metric.max()-metric.min())    
       metric_scaled = (2 * norm(metric)) - 1
       metric=metric_scaled
      
       # Set a threshold 
       metric[metric <0]=-1
        
       return metric

             
    def write_metric(self):
        
        metric=self.calculate_metric()
        
        # Skip if non_zero pixels < 0.90 threshold
        if type(metric)!=np.ndarray:
            return
        
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
        
        # Skip if non_zero pixels < 0.90 threshold
        if type(metric)!=np.ndarray:
            return 
        
        # Plot the output
        self.plot(metric,mask=0) 
        
        # Plot output with mask comparison
        self.bi_plot_mask(metric)
 

        
     