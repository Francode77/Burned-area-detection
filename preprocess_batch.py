# This file will load each image pair and mask in the folds from the source files,
# and output the metric

import os
from classes.field import Field

# Source files  
files=['train_eval.hdf5','california_0.hdf5','california_1.hdf5','california_2.hdf5']

for idx, file in enumerate(files):
    source_file=os.path.join('data',file)
    for fold in range (0,7):
        try: 
            for x in range (0,99):
                print(f'\rFold: {fold}, Image: {x}', end='', flush=True)
                try:
                    see=Field(source_file,fold,x)
                    see.write_metric() 
                except:
                    break
        except:
            break
 