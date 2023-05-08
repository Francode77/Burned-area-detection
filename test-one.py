from classes import Field
 
see=Field(2,69)
see.delta_plot(mask=0) 
"""
see.plot_rgb(0,2) 
see.plot_ndvi(0)
see.plot_gndvi(0)  # (negative) red values for burned area
see.plot_savi(0)   # Nope
see.plot_gci(0)

see.plot_bsi(0) # Positive values (green) for burned area
see.plot_avi(0)   # OK
see.plot_ndmi(0) # GOOD
see.plot_mi(0)  # OK
see.plot_nbri(0) # OK "<<
 
see.plot_NBRI_delta()
  
see.plot_ndwi(0)
see.get_water_mask()
see.delta_plot(mask=0)
see.delta_plot(mask=1)
#see.delta_rgb(0)
#see.delta_rgb(1) 
see.plot_rgb(mask=0,factor=4) 

#see.plot_watermask()
see.plot_nbri(0) 
see.delta_plot(mask=0)"""