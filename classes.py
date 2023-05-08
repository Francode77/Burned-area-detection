# Class field

import rasterio
from loader import loader
import matplotlib.pyplot as plt
from rasterio.plot import show
import numpy as np
import os
 

def norm(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min)/(band_max - band_min))

class Field:
    def __init__(self, batch_nr,img_nr):     
        self.see=loader ('./data/train_eval.hdf5',[batch_nr]) 
        self.img_nr = img_nr # image in batch
        self.batch_nr = batch_nr 
        
    def bands(self,scene_nr,band_nr):      
        self.band=self.see[scene_nr][self.img_nr,:,:,band_nr] 
        return self.band

    def get_rgb(self,scene_nr):    
        r=self.bands(scene_nr,3)
        g=self.bands(scene_nr,2)
        b=self.bands(scene_nr,1)   
        return r,g,b     
           
    def return_mask(self):
        self.mask=self.see[2][self.img_nr,:,:,0]
        return self.mask   
    
    def plot(self,indicator,mask):
        fig, ax = plt.subplots(figsize=(10, 10))  
        cmap=None
        ax.imshow(indicator,cmap=cmap,vmin=-1,vmax=1)
        if mask==1:
            mask=self.return_mask()       
            ax.imshow(mask,alpha=.33)
            
    def plot_watermask(self):
        
        water_mask=self.get_water_mask()
        fig, ax = plt.subplots(figsize=(10, 10)) 
        ax.imshow(water_mask)
        
            
    def plot_mask(self,img): 
        
        mask=self.return_mask()       
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 15))        
 
        axs[0].imshow(img)
        axs[0].set_title(f'RESULT | Fold {self.batch_nr} | Image {self.img_nr}')        
        axs[1].imshow(mask)
        axs[1].set_title(f'MASK | Fold {self.batch_nr} | Image {self.img_nr}') 
        
    def bi_plot(self,indicator,before,after,cmap,mask):
            fig, axs = plt.subplots(1, 2, figsize=(15, 15))        
 
            axs[0].imshow(after,cmap=cmap)
            axs[0].set_title(f'{indicator} Scene 0 | Fold {self.batch_nr} | Image {self.img_nr}')    
            axs[1].imshow(before,cmap=cmap)
            if mask==1:
                mask=self.return_mask()       
                axs[1].imshow(mask,alpha=.3)
            axs[1].set_title(f'{indicator} Scene 1 | Fold {self.batch_nr} | Image {self.img_nr}') 
            
            
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
        B11=self.bands(0,11)
        #NDMI_after=(B08 - B11) / (B08 + B11)
        NDMI_after = (B08.astype(float) - B11.astype(float)) / (B08.astype(float) + B11.astype(float) + 1e-10)
        
        B08=self.bands(1,7)
        B11=self.bands(1,11)
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
        
        water_mask=-water_mask
        return water_mask

    def get_delta_positive(self,after,before):
        mean_delta=after.mean()- before.mean()
        before_norm=before-mean_delta
        delta=before_norm-after
        return delta
    
    def get_delta_negative(self,after,before):
        mean_delta=before.mean()- after.mean()
        after_norm=after-mean_delta
        delta=after_norm-before     
        return delta
     
    def delta_plot(self,mask):
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
            BSI_delta=self.get_delta_positive(BSI_after, BSI_before)
            NBRI_delta=self.get_delta_positive(NBRI_after, NBRI_before)
            
            # Fire has negative values in these indicators
            NDMI_delta=self.get_delta_negative(NDMI_after, NDMI_before)
            GNDVI_delta=self.get_delta_negative(GNDVI_after,GNDVI_before)
     
            # Sum positive deltas and inverse of negative deltas
            delta=BSI_delta + NBRI_delta - NDMI_delta - GNDVI_delta
       
        else:
            delta=BSI_after + NBRI_after - NDMI_after - GNDVI_after
            return
            
        
        # Mask water
        water_mask=self.get_water_mask()
        delta=delta*water_mask
        
        # Set a threshold 
        delta[delta <0]=-1
                
        # Save the file
        output_dir = os.path.join("data","processed") 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.savetxt(f'{output_dir}/{self.batch_nr}_{self.img_nr}.txt', delta)
        
        # Load the saved txt file
        #delta_loaded = np.loadtxt(f'{output_dir}/{self.batch_nr}_{self.img_nr}.txt')

        # Get the truth value mask        
        mask=self.return_mask()       
        
        # Save the mask
        output_dir = os.path.join("data","processed","truth") 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.savetxt(f'{output_dir}/{self.batch_nr}_{self.img_nr}.txt', mask)
        
        # Plot the output
        #self.plot(delta,mask=mask) 
        
        # Plot output with mask comparison
        #self.plot_mask(delta_loaded)
        
        
        
 

        
     