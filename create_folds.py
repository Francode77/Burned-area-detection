import os 
from loader import create_folds

files=['california_0.hdf5','california_1.hdf5','california_2.hdf5']

for file in files:
    source=os.path.join('data',file)
    create_folds(source,66) 
    