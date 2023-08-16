from classes.field import Field 
from classes.datasource import DataSource
from classes.plotters import FieldPlotter, Plotter
 
file_name = './data/train_eval.hdf5'
fold = 0
x = 5

# Load post-fire image from datasource
image = DataSource.get_image(file_name, fold, x, 0)

# Load object with uuid from datasource
see = Field(file_name, fold, x)
field_plotter = FieldPlotter(see)

Plotter.plot_rgb(image=image,brightness=2.3) # Plot the RGB

field_plotter.bi_plot_ndvi(0)   # Normalized Difference Vegetation
field_plotter.bi_plot_gndvi(0)  # Green Normalized Difference Vegetation
field_plotter.bi_plot_savi(0)   # Soil Adjusted Vegetation
field_plotter.bi_plot_gci(0)    # Green Coverage Index
field_plotter.bi_plot_bsi(0)    # Bare Soil Index
field_plotter.bi_plot_avi(0)    # Advanced Vegetation Inde
field_plotter.bi_plot_ndmi(0)   # Normalized Difference Moisture Index
field_plotter.bi_plot_mi(0)     # Moisture Index
field_plotter.bi_plot_nbri(0)   # Normalized Burned Ratio Index 
field_plotter.bi_plot_bai(0)    # Burned Area Index
field_plotter.bi_plot_abai(0)   # Analytical Burned Area Index

field_plotter.bi_plot_NBRI_delta() # Post-fire - pre-fire NBRI

field_plotter.plot_watermask() 
field_plotter.plot_firemask()

field_plotter.plot_metric(0)    # Plot our metric
