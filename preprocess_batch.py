from field import Field
import os

# Fold 0 : 78
# Fold 1 : 55
# Fold 2 : 69
# Fold 3 : 85
# Fold 4 : 69
files=['train_eval.hdf5'
#       ,'california_0.hdf5','california_1.hdf5','california_2.hdf5'
       ]
fold_len=[78,55,69,85,69]
for file in files:
    source_file=os.path.join('data',file)
    for fold in range (1,5):
        for x in range (0,fold_len[fold]):
            print (fold, x)
            see=Field(source_file,fold,x)
            see.write_metric() 