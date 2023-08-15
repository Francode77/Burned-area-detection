from classes.datasource import DataSource
from classes.field import Field
from classes.plotters import FieldPlotter, Plotter
import os 

files=['train_eval.hdf5','california_0.hdf5','california_1.hdf5','california_2.hdf5','california_4.hdf5']
 
for idx, file in enumerate(files):
    source_file=os.path.join('data',file) 
    try:
        for fold in range (2,7):
            try:
                for x in range (45,99):
                    try:
                        print (source_file, fold, x)
                        
                        # Plot the post-fire image, scene 0
                        image = DataSource.get_image(source_file, fold, x, 0)        
                        Plotter.plot_rgb(image=image, brightness=2)
                        
                        # Load the image pair and mask
                        see = Field(source_file, fold, x)
                        
                        # Load the object into the plotter
                        field_plotter = FieldPlotter(see)
                        
                        # Plot the fire detection and metric
                        field_plotter.plot_fire()
                        field_plotter.plot_metric(1)
                        
                        input("Press Enter to continue...")
                        
                    except:
                        break
            except:
                break 
    except: 
        break 