from classes import Field
 
# Fold 0 : 78
# Fold 1 : 55
# Fold 2 : 69
# Fold 3 : 85
# Fold 4 : 69

fold_len=[78,55,69,85,69]
for fold in range (4,5):
    for x in range (27,fold_len[fold]):
        print (fold, x)
        see=Field(fold,x)
        see.plot_rgb(0,2)
        see.plot_metric(1)
        input("Press Enter to continue...")