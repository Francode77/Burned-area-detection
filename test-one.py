from field import Field , DataSource
from plotting import FieldPlotter, Plotter
from make_predictions import MakePrediction
import os

files=['train_eval.hdf5'
#       ,'california_0.hdf5','california_1.hdf5','california_2.hdf5'
       ]
MODEL_NAME = 'resnet101'
MODEL_NAME = 'resnet50'
brightness = 2

fold = 0
x = 69

source_file=files[0]
source_file=os.path.join('data',source_file)
field=Field(source_file,fold,x)   
 
see=FieldPlotter(field)

see.plot_metric(mask=0)  


#Plotter(DataSource.get_image('./data/train_eval.hdf5', 4, 27, 0)).plot_rgb(brightness=2)

see.bi_plot_ndvi(0)
see.bi_plot_gndvi(0)  # (negative) red values for burned area
see.bi_plot_savi(0)   # Nope
see.bi_plot_gci(0)

see.bi_plot_bsi(0) # Positive values (green) for burned area
see.bi_plot_avi(0)   # OK
see.bi_plot_ndmi(0) # GOOD
see.bi_plot_mi(0)  # OK
see.bi_plot_nbri(0) # OK "<<
see.bi_plot_bai(0) # ???

see.bi_plot_NBRI_delta()

"""
#see.bi_plot_abai(0)

image = DataSource.get_image(source_file,fold, x, 0)  
mask_truth = DataSource.get_mask(source_file, fold, x)
metric, prediction, mask_pred = MakePrediction.predict(image, MODEL_NAME) 

Plotter.plot_rgb(image, brightness) 

see.plot_fire()
see.plot_firemask(field)
see.plot_mask(prediction)
see.plot_mask(mask_truth)

"""
see.bi_plot_ndwi(0)

"""
#see.get_water_mask()
#see.delta_rgb(0)
#see.delta_rgb(1) 
#see.plot_rgb(mask=0,factor=4) 
#see.plot_watermask()"""
