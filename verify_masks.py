import os
import sys
import rasterio
from skimage.transform import resize
import torch
import segmentation_models_pytorch as smp 
from numpy.typing import NDArray
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import pandas as pd
import keyboard

def bi_plot(img, preds, truth, filename):

    fig, axs = plt.subplots(1, 2, figsize=(15, 15))
    cmap=None
    axs[0].imshow(img, cmap=cmap)
    axs[0].set_title(f'Metric {filename}')
    #axs[1].imshow(preds, cmap=cmap)
    #axs[1].set_title('Predicted mask')
    axs[1].imshow(truth, cmap=cmap) 
    axs[1].set_title('Truth mask')
    plt.show()

class VerifyPrediction:
        
    def predict(image : NDArray(512)) -> NDArray:
        IMAGE_HEIGHT= 256
        DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
        model_name  = "xception"
        MODEL_PATH  = f"{model_name}_trained_{IMAGE_HEIGHT}px.pt"
        
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

def process(df,filename):
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
            
    print (f'{IoU:.2f} % IoU')
    
    #input("Press Enter to continue...")
    print ('Current value: ', df.loc[df['file'] == file, 'value'].values)
            
    # Wait for a key press
    key = keyboard.read_key()
    if key == 'q':
        df.to_csv('labels_verify.csv',index=False)
        sys.exit()
    if key == '1':
        value = 1
    elif key == '0':
        value = 0
    else:
        value = 3
        
    if value == 0 or value == 1:

        # Check if the column value exists
        if file not in df['file'].values:
            # Create a new row
            new_row = {'file': file, 'value': value}
        
            # Append the new row to the DataFrame
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        else:
            # Update the existing row
            df.loc[df['file'] == file, 'value'] = value
    elif value == 3:
        pass
    return df

directory = 'data/processed/training_scene'
#df=pd.DataFrame(columns=['file','value'])
df=pd.read_csv('labels_verify.csv',sep=',')

key='dummy'
print ('Press A for ALL E for EMPTY or 0 for suspicious')
while key != 'a' or key !='A' or key !='0':
    key = keyboard.read_key()
    if key == 'a' or key == 'A':
        selection = 'all'
        break
    elif key == '0':
        selection = 'suspicious'
        break
    
    elif key == 'e' or key == 'E':
        selection = 'empty'
        break

print ('Press 0 for incorrect label or ENTER')
       
for root, dirs, files in os.walk(directory):
    for file in files:
        filename = os.path.join(root, file)
        
        if selection == 'suspicious':
        
            if df.loc[df['file'] == file, 'value'].values == 0:
                
                df = process(df,filename)                
            
            else: 
                pass
                
        elif selection == 'all':
            df = process(df,filename)
        
        else:
            if file not in df['file'].values:
                df = process(df,filename)
            else:
                pass
            
                            

df.to_csv('labels_verify.csv',index=False)