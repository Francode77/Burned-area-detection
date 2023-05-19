from field import Field
import os
"""
# 'train_eval.hdf5'
source_file=os.path.join('data','train_eval.hdf5')
for fold in range (1,5):
    for x in range (0,99):
        print (fold, x)
        try: 
            see=Field(source_file,fold,x)
            see.write_metric() 
        except:
            break
"""
# Extra data  
fold_lengths=[6,6,1]
files=['california_0.hdf5','california_1.hdf5','california_2.hdf5']
for idx, file in enumerate(files):
    source_file=os.path.join('data',file)
    for fold in range (0,7):
        try: 
            for x in range (0,99):
                print (file, fold, x)
                try:
                    see=Field(source_file,fold,x)
                    see.write_metric() 
                except:
                    break
        except:
            break
