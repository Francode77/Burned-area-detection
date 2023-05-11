import os
import rasterio
from skimage.transform import resize
import torch
import segmentation_models_pytorch as smp 
from numpy.typing import NDArray
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt

def bi_plot(img, preds, truth, filename):

    fig, axs = plt.subplots(1, 3, figsize=(15, 15))
    cmap=None
    axs[0].imshow(img, cmap=cmap)
    axs[0].set_title(f'Metric {filename}')
    axs[1].imshow(preds, cmap=cmap)
    axs[1].set_title('Predicted mask')
    axs[2].imshow(truth, cmap=cmap) 
    axs[2].set_title('Truth mask')
    plt.show()

class VerifyPrediction:
        
    def predict(image : NDArray(512)) -> NDArray:
 
        DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
        model_name  = "xception"
        MODEL_PATH  = f"{model_name}_trained.pt"
        
        model = smp.Unet(encoder_name=model_name, in_channels=1, classes=1, activation=None).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval() 
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
    

def get_iou(preds, mask): 
    
    mask = resize(mask, (128, 128))
 
    intersection = (preds * mask).sum()
    union = (preds + mask).sum() - intersection
    iou_score = intersection / (union + 1e-7)

    return iou_score

file='1_8.tif'

directory = 'data/processed/training_scene'

for root, dirs, files in os.walk(directory):
    for file in files:
        filename = os.path.join(root, file)
        
            
        with rasterio.open(filename) as src:
            # read all raster bands into a single ndarray
            image = src.read(1)
            
        predicted_mask=VerifyPrediction.predict(image) 
        
         
        filename=os.path.join('data','processed','training_truth',file)
        
        with rasterio.open(filename) as src:
            # read all raster bands into a single ndarray
            mask = src.read(1)
        
        bi_plot(image,predicted_mask,mask,file)
        IoU = get_iou (predicted_mask,mask) 
                
        print (IoU)