from field import Field , DataSource
from plotting import FieldPlotter, Plotter
import os

# Fold 0 : 78
# Fold 1 : 55
# Fold 2 : 69
# Fold 3 : 85
# Fold 4 : 69
"""
file_name = './data/train_eval.hdf5'

fold_len=[78,55,69,85,69]
for fold in range (1):
    for x in range (0,fold_len[fold]):
        
        print (fold, x)
        
        see=Field(file_name, fold, x)
        field_plotter = FieldPlotter(see)
        
        image = DataSource.get_image(file_name, fold, x, 0)
        
        Plotter.plot_rgb(image=image,brightness=2)
        field_plotter.plot_fire()
        field_plotter.plot_metric(1)
        
        input("Press Enter to continue...")
"""     
# Extra data  
fold_lengths=[6,6,1]
files=['california_0.hdf5','california_1.hdf5','california_2.hdf5']
files=['california_4.hdf5']
for idx, file in enumerate(files):
    source_file=os.path.join('data',file)
    try:
        for fold in range (0,7):
            try:
                for x in range (0,99):
                    try:
                        print (fold, x)
                        
                        see=Field(source_file, fold, x)
                        field_plotter = FieldPlotter(see)
                        
                        image = DataSource.get_image(source_file, fold, x, 0)
                        
                        Plotter.plot_rgb(image=image,brightness=2)
                        field_plotter.plot_fire()
                        field_plotter.plot_metric(1)
                        
                        input("Press Enter to continue...")
                    except:
                        break
            except:
                break 
    except: 
        break