import os
from make_predictions import MakePrediction
from plotting import FieldPlotter, Plotter
import rasterio
from field import Field, DataSource
import os 



import matplotlib.pyplot as plt

MODEL_NAME = 'xception'
MODEL_NAME = 'deeplabv3_resnet50'
MODEL_NAME = 'resnet101'
MODEL_NAME = 'resnet50'

BRIGHTNESS = 4
 
# 'train_eval.hdf5'
source_file=os.path.join('data','train_eval.hdf5')
for fold in range (2,6):
    for x in range (45,79):
        try: 
            print (fold, x)
            
            image = DataSource.get_image(source_file,fold, x, 0)             
            mask_truth = DataSource.get_mask(source_file, fold, x)
            metric, prediction, mask_pred = MakePrediction.predict(image, MODEL_NAME)  
            
            
            Plotter.plot_rgb(image, BRIGHTNESS)
            """
            field=Field(source_file,fold,x)   
             
            see=FieldPlotter(field)
            see.plot_fire()
            fig, ax = plt.subplots(figsize=(10, 10))
            cmap = None
            ax.imshow(metric, cmap=cmap)
            plt.show()
            see.plot_active_fire_mask(field)
            see.plot_mask(prediction)
            see.plot_mask(mask_truth)
            """
            

            #see=Field(source_file, fold, x)
            #plotter=FieldPlotter(see)
            #Ðplotter.plot_watermask()
            FieldPlotter.plot_evaluation(image, metric, prediction, mask_truth, BRIGHTNESS) 
            
            #Plotter.plot_rgb(image, BRIGHTNESS)
            
            
            field=Field(source_file,fold,x)   
             
            see=FieldPlotter(field)
            see.plot_fire()
            see.plot_firemask(field)
            see.plot_mask(prediction)
            see.plot_mask(mask_truth)
            
            
            
            input("Press Enter to continue...")   
            
        except (IndexError,ValueError):
            break
     
    