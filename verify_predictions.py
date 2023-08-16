import os
from classes.make_predictions import MakePrediction
from classes.plotters import FieldPlotter, Plotter  
from classes.datasource import DataSource  

#MODEL_NAME = 'xception'
#MODEL_NAME = 'resnet101'
#MODEL_NAME = 'deeplabv3_resnet50'
MODEL_NAME = 'resnet50'

BRIGHTNESS = 2.3
 
# 'train_eval.hdf5'
source_file=os.path.join('data','train_eval.hdf5')
for fold in range (0,6):
    for x in range (0,79):
        try: 
            print (fold, x)
            
            image = DataSource.get_image(source_file,fold, x, 0)             
            mask_truth = DataSource.get_mask(source_file, fold, x)
            metric, prediction, mask_pred = MakePrediction.predict(image, MODEL_NAME)  
           
            Plotter.plot_rgb(image, BRIGHTNESS)
            FieldPlotter.plot_evaluation(image, metric, prediction, mask_truth, BRIGHTNESS) 
           
            input("Press Enter to continue...")   
            
        except (IndexError,ValueError):
            break
     
    