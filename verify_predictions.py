import os
from make_predictions import MakePrediction
from plotting import FieldPlotter
import rasterio
from field import Field, DataSource
import os
 
# 'train_eval.hdf5'
source_file=os.path.join('data','train_eval.hdf5')
for fold in range (1):
    for x in range (1,2):
        try: 
            print (fold, x)
            
            image=DataSource.get_image(source_file,fold, x, 0)
 
            metric, mask, bool_output=MakePrediction.predict(image) 
 
            FieldPlotter.plot_submission(image, metric, mask, 2)    
        except IndexError:
            break
     
    