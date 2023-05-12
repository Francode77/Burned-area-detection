from field import Field
from plotting import FieldPlotter

# Fold 0 : 78
# Fold 1 : 55
# Fold 2 : 69
# Fold 3 : 85
# Fold 4 : 69

file_name = './data/train_eval.hdf5'

fold_len=[78,55,69,85,69]
for fold in range (1):
    for x in range (0,fold_len[fold]):
        print (fold, x)
        see=Field(file_name, fold, x)
        plotter = FieldPlotter(see)
        plotter.plot_rgb(0,2)
        plotter.plot_metric(1)
        input("Press Enter to continue...")