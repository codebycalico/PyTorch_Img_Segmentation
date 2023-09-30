import sys
sys.path.append('/content/Human-Segmentation-Dataset-master')

import torch
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import helper

# set up the configurations
# can think of these are the 'remote control' for the program
# change the values to see how it affects the training of the machine

CSV_FILE = '/content/Human-Segmentation-Dataset-master/train.csv'
DATA_DIR = '/content/'

# 'remote control' for this program

DEVICE = 'cuda'

EPOCHS = 30
LR = 0.003
IMG_SIZE = 320
BATCH_SIZE = 16

ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'

df = pd.read_csv(CSV_FILE)
df.head()

row = df.iloc[0]

image_path = row.images
mask_path = row.masks

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

ax1.set_title('IMAGE')
ax1.imshow(image)

ax2.set_title('MASK')
ax2.imshow(mask)

train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 42)

# augmentation functions 
# albumentation documentation: https://albumentations.ai/docs/

import albumentations as A

def get_train_augs():
  return A.Compose([
      A.Resize(IMG_SIZE, IMG_SIZE),
      A.HorizontalFlip(p = 0.5),
      A.VerticalFlip(p = 0.5)
  ], is_check_shapes=False)

def get_valid_augs():
  return A.Compose([
      A.Resize(IMG_SIZE, IMG_SIZE)
  ], is_check_shapes=False)

# create custom dataset

from torch.utils.data import Dataset

class SegmentationDataset(Dataset):

  def __init__(self, df, augmentations):
    self.df = df
    self.augmentations = augmentations

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]

    image_path = row.images
    mask_path = row.masks

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) #(h, w, c)
    mask = np.expand_dims(mask, axis = -1)

    if self.augmentations:
      data = self.augmentations(image = image, mask = mask)
      image = data['image']
      mask = data['mask']

    #(height, width, channel) -> (channel, height, width)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32) # transpose and specify the data type
    mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

    image = torch.Tensor(image) / 255.0
    mask = torch.round(torch.Tensor(mask) / 255.0) #round values to 0 or 1

    return image, mask

trainset = SegmentationDataset(train_df, get_train_augs())
validset = SegmentationDataset(valid_df, get_valid_augs())  

print(f"Size of Trainset : {len(trainset)}")
print(f"Size of Validset : {len(validset)}")

# index, can change value for testing
idx = 13

image, mask = trainset[idx]
helper.show_image(image, mask)

# load dataset into batches

from torch.utils.data import DataLoader

trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True)
validloader = DataLoader(validset, batch_size = BATCH_SIZE)

print(f"Total number of batches in trainloader: {len(trainloader)}")
print(f"Total number of batches in validloader: {len(validloader)}")

for image, mask in trainloader:
  break

print(f"One batch image shape: {image.shape}")
print(f"One batch mask shape: {mask.shape}")

# create segmentation model
# segmentation_models_pytorch documentation: https://smp.readthedocs.io/en/latest/

# segmentation models mostly based off of encoders and decorders
# which have their own encorder and decorder networks

from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

class SegmentationModel(nn.Module):
  def __init__(self):
    super(SegmentationModel, self).__init__()

    self.arc = smp.Unet(
        encoder_name = ENCODER,
        encoder_weights = WEIGHTS,
        in_channels = 3,
        classes = 1,
        activation = None
    )

  def forward(self, images, masks = None):
    logits = self.arc(images)

    if masks != None:
      #calculate the loss
      loss1 = DiceLoss(mode = 'binary')(logits, masks)
      loss2 = nn.BCEWithLogitsLoss()(logits, masks)
      return logits, loss1 + loss2

    # testing or inference, at that time our mask will be None and return just logits
    return logits

model = SegmentationModel()
# semi colon removes the output of all of the layers that are present in the Unet
# using the encoder
model.to(DEVICE);

# create and train validation function

# function used to train the model
def train_fn(data_loader, model, optimizer):
  model.train()
  total_loss = 0.0

  for images, masks in tqdm(data_loader):
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    # make sure that the gradients are zero
    optimizer.zero_grad()
    logits, loss = model(images, masks)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  # return average loss -> the total loss devided by the batch size, the length of the data loader
  return total_loss / len(data_loader)

def eval_fn(data_loader, model):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
      for images, masks in tqdm(data_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits, loss = model(images, masks)

        total_loss += loss.item()

    return total_loss / len(data_loader)

# train model

optimizer = torch.optim.Adam(model.parameters(), lr = LR)

# training loop
best_valid_loss = np.Inf

for i in range(EPOCHS):
  train_loss = train_fn(trainloader, model, optimizer)
  valid_loss = eval_fn(validloader, model)

  if valid_loss < best_valid_loss:
    torch.save(model.state_dict(), 'best_model.pt')
    print('SAVED MODEL')
    best_valid_loss = valid_loss

  print(f"Epoch: {i+1}, Train loss: {train_loss}, Valid loss: {valid_loss}")

# inference

idx = 2

# use valid test as test set
model.load_state_dict(torch.load('/content/best_model.pt'))

image, mask = validset[idx]

# getting image, mask from the valid set, it comes as (channel, height, width)
# here we are unsqueezing it to add another dimension, to (1, channel, height, width)
logits_mask = model(image.to(DEVICE).unsqueeze(0))
pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask > 0.5) * 1.0 # if it is greater than 0.5, round up to 1.0

helper.show_image(image, mask, pred_mask.detach().cpu().squeeze(0))
