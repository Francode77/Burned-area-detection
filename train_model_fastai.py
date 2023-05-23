import numpy as np
from fastai.data.all import *
from fastai.vision.all import *
from fastai.metrics import * 

# Run the model on the images in this path
image_path='./data/processed/training_scene' 
truth_path='./data/processed/training_truth' 

# Set the path of the image files
 
img_path = Path(image_path)
mask_path = Path(truth_path)
#Path.BASE_PATH = img_path
#img_path.ls()

fnames = get_image_files(img_path)
mask_files = get_image_files(mask_path)
print (len(fnames))
"""
dblock = DataBlock()
dsets = dblock.datasets(fnames)
print (dsets.train[0])
"""
dblock = DataBlock(get_items = get_image_files, get_y = get_image_files)
dsets = dblock.datasets(img_path, mask_path )
print (dsets.train[0])
 
msk = PILMask.create(mask_files[10])
#msk.show(figsize=(5,5), alpha=1)
print (tensor(msk).unique())

get_msk = lambda o: mask_path/f'{o.stem}{o.suffix}'

codes_array=np.array([0,1],dtype=np.uint8)
sz = msk.shape; 
print(sz)
half = tuple(int(x/2) for x in sz); half
camvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes=codes_array)),
                   get_items=get_image_files, 
                   get_y=get_msk,
                   splitter  = RandomSplitter(),
                   item_tfms = Resize(32))
                    
dls = camvid.dataloaders(img_path, bs=1)

#dls.show_batch(max_n=4, vmin=0, vmax=2, figsize=(14,10))

def acc_camvid(inp, targ):
  targ = targ.squeeze(1)
  mask = targ != 0
  return (inp.argmax(dim=1)[mask]==targ[mask]).float().mean()

opt = ranger

learn = unet_learner(dls, resnet50, metrics=acc_camvid, self_attention=True, act_cls=Mish, opt_func=opt)
print(learn.lr_find())
print(learn.summary())

"""


fnames = get_image_files(image_path)
truth_masks= get_image_files(truth_path)
dblock = DataBlock()
dsets = dblock.datasets(fnames)
dsets.train[0]
dblock = DataBlock(get_items = get_image_files(image_path))
dsets = dblock.datasets(fnames)
dsets.train[0]

"""













"""
import pandas as pd
import numpy as np
import math
import os
import cv2
import gc               # library to clear cache

import timm             # library with pretrained models
 
from fastai.data.all import *
from fastai.vision.all import *
from fastai.metrics import * 

from pathlib import Path
from sklearn.model_selection import train_test_split

# Run the model on the images in this path
image_path='./data/processed/training_scene' 
truth_path='./data/processed/training_truth' 



class SegmentationDataset(Dataset):
    def __init__(self, input_dir, output_dir, is_train, transform=None):
        
        self.input_dir  = input_dir
        self.output_dir = output_dir
        self.transform  = transform
        
        if is_train == True:
            x = round(len(os.listdir(input_dir)) * .8)
            self.images = os.listdir(input_dir)[:x]
        else:
            x = round(len(os.listdir(input_dir)) * .8)
            self.images = os.listdir(input_dir)[x:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path    = os.path.join(self.input_dir, self.images[index])
        mask_path   = os.path.join(self.output_dir, self.images[index])
        img         = np.array(Image.open(img_path), dtype=np.float32)
        mask        = np.array(Image.open(mask_path), dtype=np.float32)
        
        if self.transform is not None:
            augmentations = self.transform(image=img, mask=mask)
            img   = augmentations["image"]
            mask  = augmentations["mask"]
        
        return img, mask

def get_loaders( inp_dir, mask_dir,batch_size,
			     train_transform, val_tranform ):
    
    train_ds     = SegmentationDataset( input_dir=inp_dir, output_dir=mask_dir,
                            is_train=True, transform=train_transform)

    train_loader = DataLoader( train_ds, batch_size=batch_size, shuffle=True )

    val_ds       = SegmentationDataset( input_dir=inp_dir, output_dir=mask_dir,
                            is_train=False, transform=val_transform)

    val_loader   = DataLoader( val_ds, batch_size=batch_size, shuffle=True  )

    return train_loader, val_loader


    TRAIN_INP_DIR = os.path.join('data','processed','training_scene')
    TRAIN_OUT_DIR = os.path.join('data','processed','training_truth')
    


# Class to instantiate the FastAI datablock
class datablock:
    def __init__(self,img_path,batch_size) -> None: 
        
        # Set the path of the image files
        self.img_path=img_path
        img_path = Path(self.img_path)
        Path.BASE_PATH = img_path
 
        # Define the datablock for FastAI
        dblock = DataBlock(

        # Designation of the independent and dependent variables
        blocks = (ImageBlock, CategoryBlock), 

        # To get a list of those files from img_path, get_image_files returns a list of all of the images in that path
        get_items = get_image_files, 
        
        # Split our training and validation sets randomly
        splitter = RandomSplitter(valid_pct=0.2, seed=42) ,

        # We are telling fastai what function to call to create the target in our dataset
        get_y = get_cnc_label_from_dict,

        # VISION AUGMENTATION : add image transformations

        # First resize and then add 8 transformed images to the dataset
        item_tfms=[Resize(448), DihedralItem()],

        # DATA AUGMENTATION

        # First crop and then rescale randomly
        batch_tfms = RandomResizedCrop (size=224, min_scale=0.6, max_scale=1.0)
        )
        self.dls = dblock.dataloaders(img_path, bs=batch_size,num_workers=7, pin_memory=True,device=torch.device('cuda')) # bs = batch size
        
# Class to instantiate the model 
class model:
    def __init__(self,model,metrics,batch_size, fine_tune, calculate_lr, unfreeze_weights, nr_unfreeze_layers,epochs) -> None:
        
        self.model=model
        self.metrics=metrics 

        # Clear cache and CUDA memory
        torch.cuda.empty_cache()
        gc.collect()  

        # Create a FastAI datablock from 'image_path'
        block=datablock(image_path,batch_size=batch_size)
        self.dls=block.dls

        #Instantiate learner
        print ('Instantiate learner')     
        self.learn = vision_learner(self.dls,self.model, metrics = self.metrics)  
        print(self.learn.loss_func)
        self.learn.cuda()

        # Calculate optimal learning rate
        if calculate_lr==True:
            print ('Calculating learning rate')
            self.lr_min,self.lr_steep,self.lr_valley,self.lr_slide=lrs = self.learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
            print(f"Minimum/10: {self.lr_min:.2e}\n steepest point: {self.lr_steep:.2e}\n valley point: {self.lr_valley:.2e}\n slide point: {self.lr_slide:.2e}")

            plt.savefig(f'./models/{self.model.__name__}_learning_rate.png')
            plt.show()

        else:
                    
            self.lr_min=3.98e-03*10
            self.lr_steep=2.51e-05*10
                
        if fine_tune==True: 

            # Fine-tune pretrained model
            print ("fine tune pretrained")
            self.learn.fine_tune(epochs, self.lr_valley)

        else: 

            # Unfreeze weights
            print ('Unfreeze and train')
            self.learn.unfreeze()
            
            # print (f'Unfreeze {nr_unfreeze_layers} and train')
            #self.learn.freeze_to(-3)

            # Learn with unfreezed weights
            self.learn.fit_one_cycle(epochs, self.lr_valley)

        # Plot the loss curve
        self.learn.recorder.plot_loss()
        plt.savefig(f'./models/{self.model.__name__}_losses.png')        
        plt.show()
        
        # Save the model
        self.export_model()
        
    # Function to print the models confusion matrix
    def print_confusion_matrix(self):

        interp = ClassificationInterpretation.from_learner(self.learn)
        interp.plot_confusion_matrix(figsize=(6,6), dpi=60)
 
        plt.savefig(f'./models/{self.model.__name__}_confusion_matrix.png')
        plt.show()
        self.conf_matrix = interp.confusion_matrix()
        total = sum(sum(self.conf_matrix))

        print("True Positives: {:.2f}%".format(100 * self.conf_matrix[1, 1] / total))
        print("False Positives: {:.2f}%".format(100 * self.conf_matrix[0, 1] / total))
        print("True Negatives: {:.2f}%".format(100 * self.conf_matrix[0, 0] / total))
        print("False Negatives: {:.2f}%".format(100 * self.conf_matrix[1, 0] / total))
        
        interp.most_confused
        plt.show()

    def print_architecture(self):

        print(self.learn.summary())
    
    # Fuction to save the current models weights 
    def export_model(self):
        self.learn.export(f'./models/{self.model.__name__}_cnc_binarylabels.pkl')
        
# Metrics
f1_score_multi = F1Score() 
precision=Precision()
recall=Recall()
balancedaccuracy=BalancedAccuracy()


# Metrics
metrics=[accuracy,balancedaccuracy,error_rate]

# Batch size
batch_size=8

# Fine tune the pretrained model
fine_tune=False

# Determine learning rate
calculate_lr=True

# Unfreeze weights or unfreeze only last (n) weights
unfreeze_weights=True
nr_unfreeze_layers=0

# Epochs
epochs=24

from timm.models import xception
pretrained_model = xception

model_xception=model(model=pretrained_model,metrics=metrics,batch_size=batch_size,fine_tune=fine_tune,calculate_lr=calculate_lr,unfreeze_weights=unfreeze_weights,nr_unfreeze_layers=nr_unfreeze_layers,epochs=epochs)
model_xception.print_confusion_matrix()"""