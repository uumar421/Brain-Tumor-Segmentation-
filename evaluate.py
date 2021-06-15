
import torch
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2
from model import unet_3d_Model
import os
import argparse
from dataset import SegmentationDataset
from transform import center_crop, ToTensor, image_normalization, one_hot_encoding
import torchvision
from loss import dice_coefficient as dice_coefficient_cls
from monai.losses import DiceLoss
from skimage import exposure
from training import valid
from torch.utils import data
import os
import torch
from time import time
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
from loss import dice_coefficient, multi_class_dice_coefficient, sens_spec

from training import train
from utils import validate_crop_size

from monai.losses import DiceLoss


def string_from_num(num):
    num_str = str(num)
    return '0'*(3-len(num_str)) + num_str


def threshold(x):
        return 255 if x > 0.5 else 0


def visualize(model_path, data_dir):
    # Parameters
    data_root = "../MICCAI_BraTS2020_TrainingData/"
    train_end = 328
    val_end = 369
    batch_size = 1
    number_workers = 1
    crop_size = [160, 192, 128]

    # Load a model
    model = unet_3d_Model()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(model_path)))
    model.cuda()
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dice_loss = DiceLoss()
    dice_coefficient = multi_class_dice_coefficient()
    
    directory_valid=[]
    for i in range(train_end, val_end):
        directory_valid.append((i+1, os.path.join(data_root, f'BraTS20_Training_{string_from_num(i+1)}')))
    # print(directory_valid)
    print(f"Validation data: {len(directory_valid)} samples.")

    val_transforms = torchvision.transforms.Compose([
        image_normalization(),
        center_crop(),
        ToTensor(),
        one_hot_encoding()
    ])
    transformations = transforms.Compose([intensity_shift(),
        mirror_flip(),
        image_normalization(),
        random_crop(size=crop_size),
        #center_crop(),
        ToTensor(),
        one_hot_encoding()
    ])
    
    valid_dataset = SegmentationDataset(directory_valid, transform=val_transforms)
    valid_dataloader = data.DataLoader(dataset=valid_dataset,
                                        batch_size=batch_size,
                                        num_workers=number_workers,
                                        shuffle=False
                                        )

    valid_measures=[]
    model.eval()
    total_loss = 0
    dice = [0, 0, 0]
    total_iterations = len(valid_dataloader)
    avg_sens = [0, 0, 0]
    avg_spec = [0, 0, 0]

    def f(x):
        return 1 if x > 0.5 else 0

    with torch.no_grad():
        val_start = time()
        for i, batch in enumerate(valid_dataloader):

            x_train, y_label = batch['image'], batch['label']
            x_train, y_label = x_train.to(device, dtype=torch.float), y_label.to(device, dtype=torch.float)

            output=model(x_train)
            # print(output.shape)

            d_loss = dice_loss(output, y_label)
            # ac_loss = active_contour_loss(output, y_label)
            # f_loss = focal_loss(output, y_label)
            dice_coeff = dice_coefficient(output, y_label)
            dice_coeff = dice_coeff.detach().cpu().numpy()

            # if save_folder:
            #     output_image = output.clone()
            #     output_image = output_image.detach().cpu().numpy()
            #     np.vectorize(f)(output_image)
            #     y_label_image = y_label.clone()
            #     y_label_image = y_label_image.detach().cpu().numpy()
            #     visualize_label(output_image, f'{save_folder}/{i}_output.png', 97)
            #     visualize_label(y_label_image, f'{save_folder}/{i}_label.png', 97)

            loss = d_loss.detach().cpu().numpy()
            sens, spec = sens_spec(output, y_label, class_num=3)

            total_loss += loss
            dice += dice_coeff

            avg_sens = [a + b for a, b in zip(avg_sens, sens)]
            avg_spec = [a + b for a, b in zip(avg_spec, spec)]

            print(f"Step: Validation, Iteration: {i+1}/{total_iterations} Loss: {loss}, Dice: {dice_coeff}, Sensitivity: {sens}, Specificity: {spec}")

            torch.cuda.empty_cache()

    total_loss = total_loss/len(valid_dataloader)
    dice = dice/len(valid_dataloader)
    avg_sens = [x/len(valid_dataloader) for x in avg_sens]
    avg_spec = [x/len(valid_dataloader) for x in avg_spec]
    print(f"Total Loss: {total_loss}, Dice: {dice}, Avg Sensitivity: {avg_sens}, Avg Specificity: {avg_spec}, Time taken: {time()-val_start}")

    return total_loss, dice

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate and visualize data using model")
    parser.add_argument('-model', '-m', type=str, default='best_model.hdf5')
    parser.add_argument('-data_root', '-d', type=str, default='/home/madil/Personal/Umar/brain_experiments/eval_data')

    args = parser.parse_args()
    visualize(args.model, args.data_root)