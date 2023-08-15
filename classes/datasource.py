# Class to read uuid object from source_file 
# [0] Scene 0 : Post-fire 
# [1] Scene 1 : Pre-fire
# [2] Scene 2 : Truth mask

from loader import loader

class DataSource:
   # Create empty dictionary
   loaded_in_batches = {}

   # Returns an image with the uuid
   def get_image(source_file, batch_nr, img_nr, scene_nr):
       batches_for_file = DataSource.loaded_in_batches.setdefault(source_file,{})
       
       if batch_nr not in batches_for_file:
           DataSource.loaded_in_batches[source_file][batch_nr] = loader(source_file, [batch_nr]) 
           
       return DataSource.loaded_in_batches[source_file][batch_nr][scene_nr][img_nr,:,:,:]

   # Returns a band from an image
   def get_band(source_file, batch_nr, img_nr, scene_nr, band_nr):
       
       batches_for_file=DataSource.loaded_in_batches.setdefault(source_file,{})
       
       if batch_nr not in batches_for_file:
           DataSource.loaded_in_batches[source_file][batch_nr] = loader(source_file, [batch_nr]) 
           
       return DataSource.loaded_in_batches[source_file][batch_nr][scene_nr][img_nr,:,:,band_nr]
 
   # Returns the truth mask with the uuid
   def get_mask(source_file, batch_nr, img_nr):
       
       batches_for_file=DataSource.loaded_in_batches.setdefault(source_file,{})
       
       if batch_nr not in batches_for_file:
           DataSource.loaded_in_batches[source_file][batch_nr] = loader(source_file, [batch_nr]) 
           
       return DataSource.loaded_in_batches[source_file][batch_nr][2][img_nr,:,:,0]

