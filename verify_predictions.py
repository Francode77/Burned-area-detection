import os
from make_predictions import MakePrediction
from plotting import FieldPlotter
import rasterio
from field import Field, DataSource
import os
 
# 'train_eval.hdf5'
source_file=os.path.join('data','train_eval.hdf5')
for fold in range (1):
    for x in range (99):
        try: 
            print (fold, x)
            
            image = DataSource.get_image(source_file,fold, x, 0) 
            mask_truth = DataSource.get_mask(source_file, fold, x)
            metric, prediction, mask_pred = MakePrediction.predict(image)  
            FieldPlotter.plot_img(self, image, 0)
            FieldPlotter.plot_evaluation(image, metric, prediction, mask_truth, 2) 
            input("Press Enter to continue...")   
            
        except IndexError:
            break
     
    