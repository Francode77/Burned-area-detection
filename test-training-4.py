import os
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import segmentation_models_pytorch as smp
import albumentations as A  
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(12)

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
    
TRAIN_INP_DIR = os.path.join('data','processed','training_scene')
TRAIN_OUT_DIR = os.path.join('data','processed','training_truth')
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 3e-5
BATCH_SIZE    = 8
NUM_EPOCHS    = 66
IMAGE_HEIGHT  = 128
IMAGE_WIDTH   = 128  

train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ],
)

val_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        ToTensorV2(),
    ],
)

def get_loaders( inp_dir, mask_dir,batch_size,
			     train_transform, val_tranform ):
    
    train_ds     = SegmentationDataset( input_dir=inp_dir, output_dir=mask_dir,
                            is_train=True, transform=train_transform)

    train_loader = DataLoader( train_ds, batch_size=batch_size, shuffle=True )

    val_ds       = SegmentationDataset( input_dir=inp_dir, output_dir=mask_dir,
                            is_train=False, transform=val_transform)

    val_loader   = DataLoader( val_ds, batch_size=batch_size, shuffle=True  )

    return train_loader, val_loader

train_loader, val_loader = get_loaders( TRAIN_INP_DIR, TRAIN_OUT_DIR,
                            BATCH_SIZE,  train_transform, val_transform)
inputs, masks = next(iter(train_loader))
print(inputs.shape)
print (len(train_loader),len(val_loader))
_, ax = plt.subplots(1,2)
ax[0].imshow(inputs[0].permute(1,2,0)[:,:,0])
ax[1].imshow(masks[0])

def predict_for_image(img, mask, device="cuda"):
    img = img.to(device)
    mask = mask.to(device).unsqueeze(1)
    preds = torch.sigmoid(model(img))
    preds = (preds > 0.5).float()

    num_correct = (preds == mask).sum()
    num_pixels = torch.numel(preds)
    dice_score = (2 * (preds * mask).sum()) / (
            (preds + mask).sum() + 1e-7
    )

    intersection = (preds * mask).sum()
    union = (preds + mask).sum() - intersection
    iou_score = intersection / (union + 1e-7)

    return num_correct, num_pixels, dice_score, iou_score

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    model.eval()

    with torch.no_grad():
        for img, mask in tqdm(loader):
            num_correct_img, num_pixels_img, dice_score_img, iou_score_img = predict_for_image(img, mask, device)
            num_correct += num_correct_img
            num_pixels += num_pixels_img
            dice_score += dice_score_img
            iou_score += iou_score_img

    print(
        f"Got {num_correct}/{num_pixels} with pixel accuracy {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)*100:.2f}")
    print(f"IoU score: {iou_score/len(loader)*100:.2f}")
    model.train()
    
model_name='xception'

model = smp.Unet(encoder_name=model_name, in_channels=1, classes=1, activation=None).to(DEVICE)
loss_fn   = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)

    for batch_idx, (image, mask) in enumerate(loop):
        image   = image.to(device=DEVICE)
        mask    = mask.float().unsqueeze(1).to(device=DEVICE)

        # forward
        predictions = model(image)
        loss = loss_fn(predictions, mask)

        # backward
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
#check_accuracy(val_loader, model, device=DEVICE)

for epoch in range(NUM_EPOCHS):

    print('########################## epoch: '+str(epoch))
    train_fn(train_loader, model, optimizer, loss_fn)
    
    # check accuracy
    check_accuracy(val_loader, model, device=DEVICE)
    
inputs, masks = next(iter(val_loader))
output        = ((torch.sigmoid(model(inputs.to(DEVICE)))) >0.5).float()

# Save the trained model
MODEL_PATH = f"{model_name}_trained.pt"
torch.save(model.state_dict(), MODEL_PATH)

_, ax = plt.subplots(2,3, figsize=(15,10))
for k in range(2):
    ax[k][0].imshow(inputs[k].permute(1,2,0))
    ax[k][1].imshow(output[k][0].cpu())
    ax[k][2].imshow(masks[k])

### WORST PREDICTIONS
def find_worst_predictions(loader, device):
    with torch.no_grad():
        for img, mask in tqdm(loader):
            num_correct_img, num_pixels_img, dice_score_img, iou_score_img = predict_for_image(img, mask, device)
            predictions_by_iou[img] = iou_score_img

predictions = False

if predictions:
    predictions_by_iou = {}
    find_worst_predictions(val_loader, model, DEVICE)
    find_worst_predictions(train_loader, model, DEVICE)
    print(predictions_by_iou)