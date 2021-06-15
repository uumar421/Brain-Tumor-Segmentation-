import os
import torch
from torch._C import Value
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
import argparse

from transform import intensity_shift
from transform import mirror_flip
from transform import image_normalization
from transform import random_crop
from transform import center_crop
from transform import ToTensor
from transform import one_hot_encoding

from dataset import SegmentationDataset

from model import unet_3d_Model
from model import normal_init

from loss import soft_dice_loss
from loss import active_contour_loss
from loss import focal_loss
from loss import dice_coefficient as dice_coefficient_cls

from training import train
from utils import validate_crop_size

from monai.losses import DiceLoss

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--weight_decay', '-wc', default=1e-5, type=float, help='Weight decay.')
parser.add_argument('--n_epochs', '-e', default=10, type=int, help='Epochs')
parser.add_argument('--batch_size', '-b', default=1, type=int, help='Batch size')
parser.add_argument('--number_workers', '-w', default=1, type=int, help='No of workers.')
parser.add_argument('--data_root', '-dr', default="../MICCAI_BraTS2020_TrainingData/", type=str, help='Data Root.')
parser.add_argument('--save_path', '-s', default="/results", type=str, help='Save path for graphs etc.')
parser.add_argument('--crop_size', nargs='+', type=int, default=[160, 192, 128], help='Required crop size. Must be 3 values and divisible by 32.')

args = parser.parse_args()


# Hyper parameters
initial_learning_rate = args.learning_rate
weight_decay = args.weight_decay
n_epochs = args.n_epochs
batch_size = args.batch_size
number_workers = args.number_workers
data_root = args.data_root
crop_size = args.crop_size
save_path = args.save_path

# Print hyper parameters for logging
print(f"Learning rate: {initial_learning_rate}")
print(f"Weight Decay: {weight_decay}")
print(f"No epochs: {n_epochs}")
print(f"Batch size: {batch_size}")
print(f"Number of workers: {number_workers}")
print(f"Data Root: {data_root}")
print(f"Crop size: {crop_size}")


validate_crop_size(crop_size)
# Set of transforms (Nothing defined here. All defined in another file. Only created here)

transformations = [intensity_shift(),
                   mirror_flip(),
                   image_normalization(),
                   random_crop(size=crop_size),
                   #center_crop(),
                   ToTensor(),
                   one_hot_encoding()
                   ]
val_transformations = [
    image_normalization(),
    center_crop(),
    ToTensor(),
    one_hot_encoding()
]



# Create dataset (defined somewhere else. only initialized here)

def string_from_num(num):
    num_str = str(num)
    return '0'*(3-len(num_str)) + num_str

train_end = 328
val_end = 369
# train_end = 7
# val_end = 9


directory_train=[]
for i in range(train_end):
    directory_train.append((i+1, os.path.join(data_root, f'BraTS20_Training_{string_from_num(i+1)}')))
# print(directory_train)
print(f"Training data: {len(directory_train)} samples.")

directory_valid=[]
for i in range(train_end, val_end):
    directory_valid.append((i+1, os.path.join(data_root, f'BraTS20_Training_{string_from_num(i+1)}')))
# print(directory_valid)
print(f"Validation data: {len(directory_valid)} samples.")

train_dataset = SegmentationDataset(directory_train, transform=transforms.Compose(transformations))
training_dataloader = data.DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      num_workers=number_workers,
                                      shuffle=True
                                      )

valid_dataset = SegmentationDataset(directory_valid, transform=transforms.Compose(val_transformations))
valid_dataloader = data.DataLoader(dataset=valid_dataset,
                                      batch_size=batch_size,
                                      num_workers=number_workers,
                                      shuffle=False
                                      )



# Create model (defined in another file)

model = unet_3d_Model()
model.cuda()
model.apply(normal_init)
model = nn.DataParallel(model)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=initial_learning_rate,
                            weight_decay=weight_decay)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dice_loss = DiceLoss()
active_contour_loss = active_contour_loss()
focal_loss = focal_loss()
dice_coefficient = dice_coefficient_cls()

print(f"Using {torch.cuda.device_count()} GPUs.")



# Run model (trainign loop defined in another file)

import pickle

measures_iter, measures_epoch = train(
    training_dataloader, model, device, optimizer,
    dice_loss, active_contour_loss, focal_loss,
    dice_coefficient, valid_dataloader=valid_dataloader,
    initial_learning_rate=initial_learning_rate, n_epochs=n_epochs, save_path=save_path)
try:
    with open(os.path.join(save_path, 'measures_iter.pkl'), 'wb') as f:
        pickle.dump(measures_iter, f)
    with open(os.path.join(save_path, 'measures_epoch.pkl'), 'wb') as f:
        pickle.dump(measures_epoch, f)
except Exception as e:
    print(e)
