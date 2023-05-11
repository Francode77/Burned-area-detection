from typing import Any, Union, Dict, Literal
from pathlib import Path
from numpy.typing import NDArray
from collections import defaultdict
import h5py
import torch
import segmentation_models_pytorch as smp 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt

def norm(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min)/(band_max - band_min))

def stand(band):
    band_min, band_max = band.min(), band.max()
    return ((2 * (band - band_min)/(band_max - band_min)) - 1)
    
def retrieve_validation_fold(path: Union[str, Path]) -> Dict[str, NDArray]:
    result = defaultdict(dict)
    with h5py.File(path, 'r') as fp:
        for uuid, values in fp.items():
            if values.attrs['fold'] != 0:
                continue
            
            result[uuid]['post'] = values['post_fire'][...]
            # result[uuid]['pre'] = values['pre_fire'][...]

    return dict(result)

validation_fold = retrieve_validation_fold('data/train_eval.hdf5')
image=validation_fold['06181a53-1181-427c-9f60-55040bde0a9a_0']['post']
#print (validation_fold.keys()) 


class MakePrediction:
        
    def predict(image : NDArray(512)) -> NDArray:
        
        metric = MakePrediction.calculate_metric(image)
        image=metric

        DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
        model_name  = "xception"
        MODEL_PATH  = f"{model_name}_trained.pt"
        
        model = smp.Unet(encoder_name=model_name, in_channels=1, classes=1, activation=None).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print (image.shape) 
        transform_norm = transforms.Compose([transforms.ToTensor(), 
            transforms.Resize((128,128),antialias=True)])
        # get normalized image
        img_normalized = transform_norm(image).float()
        img_normalized = img_normalized.unsqueeze_(0)
        tensor_image= img_normalized.to(DEVICE)
 
        result = ((torch.sigmoid(model(tensor_image.to(DEVICE)))) >0.5).float()
         
        image = result[0].cpu().numpy()
        image = image[0]
        bool_output = image.astype(bool)
        
        return bool_output
    
    def get_band(image, band_nr : int):
        band=image[:,:,band_nr]
        return band

    def get_ndwi(image):        
        B03=MakePrediction.get_band(image,2)
        B08=MakePrediction.get_band(image,7)         
        NDWI = (B03.astype(float) - B08.astype(float)) / (B03.astype(float) + B08.astype(float) + 1e-10)   
        return NDWI 
    
    def get_abai(image):        
        B03=MakePrediction.get_band(image,2) 
        B11=MakePrediction.get_band(image,10) 
        B12=MakePrediction.get_band(image, 11) 
        ABAI = (3*B12 - 2 * B11 - 3 * B03 ) / ((3*B12 + 2*B11 +3*B03) + 1e-10)
        return ABAI
    
    def get_water_mask(image):

        NDWI=MakePrediction.get_ndwi(image)     
        water_mask=NDWI
        water_mask[water_mask >= 0] = 1
        water_mask[water_mask < 0] = -1
        return water_mask
    
    def calculate_metric(image):
        # Get the indices 
        ABAI = MakePrediction.get_abai(image)  
        metric = ABAI
        
        # Mask water with min value of metric
        water_mask=MakePrediction.get_water_mask(image) 
        metric[water_mask == 1] = metric.min()
        
        # Normalize & standardize        
        metric_scaled = (2 * norm(metric)) - 1
        metric=metric_scaled 
        
        # Set a threshold 
        metric[metric <0]=-1
        
        return metric
     
bool_output=MakePrediction.predict(image) 
print (bool_output)
"""
image,bool_output=MakePrediction.predict(image) 
  
print (type(image))
print (image.shape)
print (image)

plt.imshow(image ,cmap=None)

bool_arr = image.astype(bool)
print (bool_arr)  """