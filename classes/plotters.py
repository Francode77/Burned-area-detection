from classes.field import Field
from classes.field import norm
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import plot
import matplotlib.colors as mcolors

class Plotter:
    def __init__(self,image):
        self.image = image
        
    def plot_rgb(image, brightness=1):        
        r = image[:,:,3]
        g = image[:,:,2]
        b = image[:,:,1]        
        RGB = rasterio.plot.reshape_as_image([norm(r), norm(g), norm(b)])
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(RGB * brightness)
        plt.show()
        
    def plot_fire(image):        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        plt.show()

# Plots from uuid objects in the hdf5 source files
class FieldPlotter:
    def __init__(self,field : Field):
        self.field = field

    """ PLOT FUNCTIONS """
    def plot_img(self, indicator : np.ndarray , mask : int):
        fig, ax = plt.subplots(figsize=(10, 10))
        cmap = None
        ax.imshow(indicator, cmap=cmap)
        
        # Shows mask on image if mask=1
        if mask == 1:
            mask = self.field.return_mask()
            ax.imshow(mask, alpha=.33)
            
        plt.show()
    
    def plot_mask(self, mask : int):        
        fig, ax = plt.subplots(figsize=(10, 10))
        cmap = mcolors.ListedColormap(['purple', 'yellow'])
        ax.imshow(mask, cmap=cmap)
        plt.show()
        
    """ METRIC VISUALISATION """
    def plot_metric(self, mask=0):
        metric = self.field.calculate_metric()

        # Plot the output
        self.plot_img(metric, mask)

        # Plot output with mask comparison
        self.bi_plot_mask(metric)    

    def plot_fire(self):
        
        r,g,b,_ = self.field.fire_detection()
        fire_RGB = rasterio.plot.reshape_as_image([norm(r), norm(g), norm(b)])
        Plotter.plot_fire(fire_RGB)
        
        
    """ BI_PLOT VISUALISATION """
    
    # Plots indicator for scene [0] & [1]
    def bi_plot(self, indicator, before, after, cmap, mask):

        fig, axs = plt.subplots(1, 2, figsize=(15, 15))
        axs[0].imshow(after, cmap=cmap)
        axs[0].set_title(f'{indicator} Scene 0 | Fold {self.field.batch_nr} | Image {self.field.img_nr}')
       
        axs[1].imshow(before, cmap=cmap)
        if mask == 1:
            mask = self.field.return_mask()
            axs[1].imshow(mask, alpha=.3)
        axs[1].set_title(f'{indicator} Scene 1 | Fold {self.field.batch_nr} | Image {self.field.img_nr}')
        
        plt.show()


    """ MASK VISUALISATION """
    # Plots image next to mask
    def bi_plot_mask(self, img : np.ndarray):
        mask = self.field.return_mask()
        fig, axs = plt.subplots(1, 2, figsize=(15, 15))
        cmap = None
        axs[0].imshow(img, cmap=cmap, vmin=-1, vmax=1)
        axs[0].set_title(f'RESULT | Fold {self.field.batch_nr} | Image {self.field.img_nr}')
        axs[1].imshow(mask)
        axs[1].set_title(f'MASK | Fold {self.field.batch_nr} | Image {self.field.img_nr}')
        plt.show()
        
    def plot_watermask(self):
        water_mask = self.field.get_water_mask()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(water_mask)
        
    def plot_firemask(self):
        mask = self.field.get_firemask() 
        cmap = mcolors.ListedColormap(['purple', 'yellow'])
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(mask, cmap=cmap)
        plt.show()       
        
    def plot_active_fire_mask(self, image : np.ndarray):
        mask = Field.active_fire_mask(image) 
        cmap = mcolors.ListedColormap(['purple', 'yellow'])
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(mask, cmap=cmap)
        plt.show()        

    """ PLOT INDICES FUNCTIONS"""
 
    def bi_plot_rgb(self, mask : int, brightness : float):
        
        r, g, b = self.field.get_rgb(0)
        RGB_after = rasterio.plot.reshape_as_image([norm(r), norm(g), norm(b)])

        r, g, b = self.field.get_rgb(1)
        RGB_before = rasterio.plot.reshape_as_image([norm(r), norm(g), norm(b)])

        RGB_before = RGB_before * brightness
        RGB_before[RGB_before>1] = 1
        
        RGB_after = RGB_after * brightness
        RGB_after[RGB_after>1] = 1        
        
        self.bi_plot("RGB", RGB_before, RGB_after, cmap=None, mask=mask)

    def bi_plot_abai(self, mask : int):

        B03 = self.field.bands(0, 2)
        B11 = self.field.bands(0, 10)
        B12 = self.field.bands(0, 11)
        ABAI_after = (3 * B12 - 2 * B11 - 3 * B03) / ((3 * B12 + 2 * B11 + 3 * B03) + 1e-10)
        
        B03 = self.field.bands(1, 2)
        B11 = self.field.bands(1, 10)
        B12 = self.field.bands(1, 11)
        ABAI_before = (3 * B12 - 2 * B11 - 3 * B03) / ((3 * B12 + 2 * B11 + 3 * B03) + 1e-10)

        cmap = None
        self.bi_plot("ABAI", ABAI_before, ABAI_after, cmap=cmap, mask=mask)
 
    
    """ PLOT EVALUATION FUNCTIONS """
    
    def plot_evaluation(image, metric, pred_mask, truth, brightness):
        
        cmap = mcolors.ListedColormap(['purple', 'yellow'])
        r = image[:,:,3]
        g = image[:,:,2]
        b = image[:,:,1]
        rgb_image = rasterio.plot.reshape_as_image([norm(r), norm(g), norm(b)])
        
        B03 = image[:,:,2]
        B08 = image[:,:,7]
        NDWI = (B03.astype(float) - B08.astype(float)) / (B03.astype(float) + B08.astype(float) + 1e-10)   
    
        NDWI[NDWI>0] = 1
        NDWI[NDWI<0] = -1
        
        rgb_image = rgb_image * brightness
        rgb_image[rgb_image>1] = 1
        
        fig, axs = plt.subplots(1, 4, figsize=(15, 15))
        axs[0].imshow((rgb_image), cmap=None)
        axs[0].set_title('RGB') 
        
        axs[1].imshow(metric, cmap=None)
        axs[1].set_title('METRIC')
        
        axs[2].imshow(pred_mask, cmap=cmap)
        axs[2].set_title('Our MASK')
        
        axs[3].imshow(truth, cmap=cmap)
        axs[3].set_title('TRUTH')
        
        plt.show()
        
    
    def plot_submission(image, metric, pred_mask, brightness):
        
        cmap = mcolors.ListedColormap(['purple', 'yellow'])
        r = image[:,:,3]
        g = image[:,:,2]
        b = image[:,:,1]
        rgb_image = rasterio.plot.reshape_as_image([norm(r), norm(g), norm(b)])
        
        B03 = image[:,:,2]
        B08 = image[:,:,7]
        NDWI = (B03.astype(float) - B08.astype(float)) / (B03.astype(float) + B08.astype(float) + 1e-10)   
        NDWI[NDWI>0] = 1
        NDWI[NDWI<0] = -1
        
        rgb_image = rgb_image * brightness
        rgb_image[rgb_image>1] = 1
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 15))
        axs[0].imshow((rgb_image), cmap=None)
        axs[0].set_title('RGB') 
        axs[1].imshow(metric, cmap=None)
        axs[1].set_title('METRIC')
        axs[2].imshow(pred_mask, cmap=cmap)
        axs[2].set_title('Our MASK')
        
        plt.show()
        
        
        
    ### BI_PLOTS for ALL INDEXES    
                              
    def bi_plot_ndvi(self, mask : int):
        r_before = self.field.bands(1,3).astype(float)
        nir_before = self.field.bands(1,7).astype(float)
        NDVI_before = (nir_before - r_before) / ((nir_before + r_before) + 1e-10)
   
        r_after = self.field.bands(0,3).astype(float)
        nir_after = self.field.bands(0,7).astype(float)
        NDVI_after = (nir_after - r_after) / ((nir_after + r_after) + 1e-10)
        
        cmap = 'RdYlGn'
        self.bi_plot("NDVI",NDVI_before,NDVI_after,cmap=cmap,mask=mask)
        
    def bi_plot_gndvi(self, mask : int):
        
        B03 = self.field.bands(1,2).astype(float)
        B08 = self.field.bands(1,7).astype(float)
        GNDVI_before = (B08 - B03) / ((B08 + B03) + 1e-10)
        B03 = self.field.bands(0,2).astype(float)
        B08 = self.field.bands(0,7).astype(float)
        GNDVI_after = (B08 - B03) / ((B08 + B03) + 1e-10)
        
        cmap = 'RdYlGn'
        self.bi_plot("GNDVI",GNDVI_before,GNDVI_after,cmap=cmap,mask=mask)
        
    def bi_plot_bsi(self, mask :int):
        B02 = self.field.bands(0,1).astype(float)
        B04 = self.field.bands(0,3).astype(float)
        B08 = self.field.bands(0,7).astype(float)
        B11 = self.field.bands(0,11).astype(float)
        BSI_after = ((B11 + B04) - (B08 + B02)) / (((B11 + B04) + (B08 + B02))+ 1e-10)
 
        B02 = self.field.bands(1,1).astype(float)
        B04 = self.field.bands(1,3).astype(float)
        B08 = self.field.bands(1,7).astype(float)
        B11 = self.field.bands(1,11).astype(float)
        BSI_before = ((B11 + B04) - (B08 + B02)) / (((B11 + B04) + (B08 + B02))+ 1e-10)

        cmap = 'RdYlGn'
        self.bi_plot("BSI",BSI_before,BSI_after,cmap=cmap,mask=mask)
        
    def bi_plot_mi(self ,mask : int):   
        B8A = self.field.bands(1,8).astype(float)
        B11 = self.field.bands(1,10).astype(float)
        MI_before = (B8A - B11) / ((B8A+B11) + 1e-10)
        B8A = self.field.bands(0,8).astype(float)
        B11 = self.field.bands(0,10).astype(float)
        MI_after = (B8A - B11) / ((B8A+B11) + 1e-10)
 
        cmap='RdYlGn'
        self.bi_plot("MI",MI_before,MI_after,cmap=cmap,mask=mask)
        
    def bi_plot_avi(self, mask : int):  
        
        B04 = self.field.bands(1,3).astype(float)
        B08 = self.field.bands(1,7).astype(float)
        AVI_before = (B08 * (1 - B04)*(B08 - B04))**1/3
        
        B04 = self.field.bands(0,3).astype(float)
        B08 = self.field.bands(0,7).astype(float)
        AVI_after = (B08 * (1 - B04)*(B08 - B04))**1/3
        
        cmap='RdYlGn'
        self.bi_plot("AVI",AVI_before,AVI_after,cmap=cmap,mask=mask)
        
    def bi_plot_savi(self, mask : int):  
        
        B04 = self.field.bands(1,3).astype(float)
        B08 = self.field.bands(1,7).astype(float)
        SAVI_before = (B08 - B04) / (B08 + B04 + 0.428) * (1.428)
        
        B04 = self.field.bands(0,3).astype(float)
        B08 = self.field.bands(0,7).astype(float)
        SAVI_after = (B08 - B04) / (B08 + B04 + 0.428) * (1.428)
        
        cmap = 'RdYlGn'
        self.bi_plot("SAVI",SAVI_before,SAVI_after,cmap=cmap,mask=mask)
        
    def bi_plot_ndmi(self, mask : int):  
        
        B08 = self.field.bands(0,7)
        B11 = self.field.bands(0,11)
        NDMI_after = (B08.astype(float) - B11.astype(float)) / ((B08.astype(float) + B11.astype(float))+ 1e-10)
        
        B08 = self.field.bands(1,7)
        B11 = self.field.bands(1,11)
        NDMI_before = (B08.astype(float) - B11.astype(float)) / ((B08.astype(float) + B11.astype(float))+ 1e-10)
 
        cmap = 'RdYlGn'
        self.bi_plot("NDMI",NDMI_before,NDMI_after,cmap=cmap,mask=mask)
        
    def bi_plot_gci(self, mask : int):  
        
        B03 = self.field.bands(0,2)
        B09 = self.field.bands(0,9)
        GCI_after = (B09.astype(float) / (B03.astype(float) + 1e-10)) -1
        
        B03 = self.field.bands(1,2)
        B09 = self.field.bands(1,9)
        GCI_before = (B09.astype(float) / (B03.astype(float) + 1e-10)) -1
        
        cmap = 'RdYlGn'
        self.bi_plot("GCI",GCI_before,GCI_after,cmap=cmap,mask=mask)
             
    def bi_plot_nbri(self, mask : int):  
        
        B08 = self.field.bands(0,7)
        B12 = self.field.bands(0,11)
        NBRI_after = (B08.astype(float) - B12.astype(float)) /((B08.astype(float) + B12.astype(float) + 1e-10))

        B08 = self.field.bands(1,7)
        B12 = self.field.bands(1,11)
        NBRI_before = (B08.astype(float) - B12.astype(float)) / ((B08.astype(float) + B12.astype(float) + 1e-10))

        cmap = None
        self.bi_plot("NBRI",NBRI_before,NBRI_after,cmap=cmap,mask=mask)
          
    def bi_plot_ndwi(self, mask : int):  
        
        B03 = self.field.bands(0,2)
        B08 = self.field.bands(0,7)        
        NDWI_after = (B03.astype(float) - B08.astype(float)) / (B03.astype(float) + B08.astype(float) + 1e-10)   
        
        B03 = self.field.bands(1,2)
        B08 = self.field.bands(1,7)
        NDWI_before = (B03.astype(float) - B08.astype(float)) / (B03.astype(float) + B08.astype(float) + 1e-10)   
        
        cmap=None
        self.bi_plot("NDWI",NDWI_before,NDWI_after,cmap=cmap,mask=mask)
 
    def bi_plot_bai(self, mask : int):  

        RED = self.field.bands(0,3) /10000 # Red 
        NIR = self.field.bands(0,7) /10000 # NIR
        
        BAI_after = 1/((0.1 -RED)**2 + (0.06 - NIR)**2)
        
        RED = self.field.bands(1,3) /10000 # Red
        NIR = self.field.bands(1,7) /10000 # NIR
        
        BAI_before = 1/((0.1 -RED)**2 + (0.06 - NIR)**2) 
        
        print (RED.mean(),BAI_after.mean())
        cmap = None
        self.bi_plot("BAI", BAI_before, norm(BAI_after),cmap=cmap,mask=mask)
    
 
    def bi_plot_NBRI_delta(self):
        
        B08 = self.field.bands(0,7).astype(float)
        B12 = self.field.bands(0,11).astype(float)
        NBRI_after = (B08 - B12) / ((B08 + B12)+ 1e-10)
        
        B08 = self.field.bands(1,7).astype(float)
        B12 = self.field.bands(1,11).astype(float)
        NBRI_before = (B08 - B12) / ((B08 + B12)+ 1e-10)
        
        NBRI_delta = NBRI_before - NBRI_after
        
        mean_delta = NBRI_after.mean()- NBRI_before.mean()
        NBRI_after_norm = NBRI_after- mean_delta
        NBRI_before_norm = NBRI_before
             
        NBRI_delta_norm = NBRI_before_norm - NBRI_after_norm
            
        fig, ax = plt.subplots(figsize=(10, 10))  
        cmap = None
        ax.imshow(NBRI_delta,cmap=cmap)
        
        fig, ax = plt.subplots(figsize=(10, 10))  
        cmap = 'RdYlGn'
        ax.imshow(NBRI_delta_norm,cmap=cmap)


        
        