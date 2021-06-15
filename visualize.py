
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


def threshold(x):
        return 255 if x > 0.5 else 0

def visualize_image_channels_and_label(image, label, plane, output_path, max_value=1, blend=0.7) :
    fig, ax = plt.subplots(3, 5, figsize=[16*3,15])
    plane_x, plane_y, plane_z = plane
    fig.tight_layout()
    for i in range(3):
        for j in range(0, 4):
            if i == 0:
                image_slice = image[0, j, plane_x, :, :]
                label_slice = label[0,:,plane_x,:,:]
            elif i == 1:
                image_slice = image[0, j, :, plane_y, :]
                label_slice = label[0,:,:,plane_y,:]
            elif i == 2:
                image_slice = image[0, j, :, :, plane_z]
                label_slice = label[0,:,:,:,plane_z]
            ax[i, j].imshow(image_slice, cmap='gray')
            ax[i, j].axis('off')
            if j == 0:
                ax[i, 4].imshow(image_slice, cmap='gray')
                for k in range(3):
                    mask = label_slice[k,:,:]
                    zeros_arr = np.zeros_like(mask)
                    alpha = np.zeros_like(mask)
                    # Get alpha blending values
                    alpha[mask==max_value] = max_value*blend

                    # Convert to color
                    if k == 0:
                        mask = np.stack([mask, zeros_arr, zeros_arr, alpha], 2)
                    elif k == 1:
                        mask = np.stack([zeros_arr, mask, zeros_arr, alpha], 2)
                    elif k == 2:
                        mask = np.stack([zeros_arr, zeros_arr, mask, alpha], 2)
                    
                    ax[i, 4].imshow(mask)
                ax[i, 4].axis('off')
    plt.plot()
    plt.savefig(output_path)


def visualize_image_and_label_planes(image, label, output_path, slice, modality, max_value=1, blend=0.7):
    slice_x, slice_y, slice_z = slice
    fig,ax=plt.subplots(1,3,figsize=[16, 9])
    for i in range(0, 3):
        if i == 0:
            brain_slice = image[0,modality,:,:,slice_z]
            label_slice = label[0,:,:,:,slice_z]
        elif i == 1:
            brain_slice = image[0,modality,slice_x,:,:]
            label_slice = label[0,:,slice_x,:,:]
        elif i == 2:
            brain_slice = image[0,modality,:,slice_y,:]
            label_slice = label[0,:,:,slice_y,:]

        print(f"Brain slice in vis: {brain_slice.shape}")
        print(f"Label slice in vis: {label_slice.shape}")
        ax[i].imshow(brain_slice, cmap="gray")
        for j in range(3):
            mask = label_slice[j,:,:]
            zeros_arr = np.zeros_like(mask)
            alpha = np.zeros_like(mask)
            # Get alpha blending values
            alpha[mask==max_value] = max_value*blend

            # Convert to color
            if j == 0:
                mask = np.stack([mask, zeros_arr, zeros_arr, alpha], 2)
            elif j == 1:
                mask = np.stack([zeros_arr, mask, zeros_arr, alpha], 2)
            elif j == 2:
                mask = np.stack([zeros_arr, zeros_arr, mask, alpha], 2)
            
            ax[i].imshow(mask)
        ax[i].axis('off')
    plt.plot()
    plt.savefig(output_path)

def visualize(model_path, data_dir):
    # Load a model
    model = unet_3d_Model()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(model_path)))
    model.cuda()
    model.eval()
    dice_loss = DiceLoss()
    dice_coefficient = dice_coefficient_cls()

    # Load data
    image_files = data_dir + '/BraTS20_Training_001_flair.nii.gz', data_dir + '/BraTS20_Training_001_t1.nii.gz', data_dir + '/BraTS20_Training_001_t1ce.nii.gz', data_dir + '/BraTS20_Training_001_t2.nii.gz'
    label_file = data_dir + '/BraTS20_Training_001_seg.nii.gz'

    image = np.array([np.array(nib.load(fname).get_fdata()) for fname in image_files])
    label = np.array(nib.load(label_file).get_fdata())

    print(f"Loaded image with shape {image.shape}")
    print(f"Loaded label with shape {label.shape}")

    data = {"image": image, "label": label}

    # viz_transforms = torchvision.transforms.Compose([
    #     center_crop(),
    #     ToTensor(),
    #     one_hot_encoding()
    # ])

    val_transforms = torchvision.transforms.Compose([
        image_normalization(),
        center_crop(),
        ToTensor(),
        one_hot_encoding()
    ])
    val_data = val_transforms(data)
    # viz_data = viz_transforms(data)
    for key in val_data:
        # viz_data[key] = torch.unsqueeze(viz_data[key], 0).float().cuda()
        val_data[key] = torch.unsqueeze(val_data[key], 0).float().cuda()

    print(val_data['image'].shape)
    print(val_data['label'].shape)

    with torch.no_grad():
        image_pt = val_data['image']
        pred = model(image_pt)

        d_loss = dice_loss(pred, val_data['label'])
        dice_coeff = dice_coefficient(pred, val_data['label'])

        pred_image = np.vectorize(threshold)(pred.cpu().numpy())
    
    
    
    print(f"Prediction output is {pred.shape}")
    print(f"Dice loss is {d_loss}")
    print(f"Dice coefficient is {dice_coeff}")

    image_np = image_pt.detach().cpu().numpy()
    label_np = val_data['label'].detach().cpu().numpy()

    np.save('pred_image', pred_image)
    np.save('image_np', image_np)
    np.save('label_np', label_np)

    slice=[66, 85, 50]
    # slice = [80, 80, 80]
    suffix = f"_{slice[0]}_{slice[1]}_{slice[2]}"
    prefix = 'results/'
    visualize_image_and_label_planes(
        image=image_np,
        label=label_np,
        output_path=f'{prefix}label_viz{suffix}.png',
        slice=slice,
        modality=0,
        max_value=1
    )

    visualize_image_and_label_planes(
        image=image_np,
        label=pred_image,
        output_path=f'{prefix}pred_viz{suffix}.png',
        slice=slice,
        modality=0,
        max_value=255
    )

    visualize_image_channels_and_label(
        image=image_np,
        label=label_np,
        plane=slice,
        output_path=f'{prefix}image_channels_and_label{suffix}.png',
        max_value=1
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate and visualize data using model")
    parser.add_argument('-model', '-m', type=str, default='best_model.hdf5')
    parser.add_argument('-data_root', '-d', type=str, default='/home/madil/Personal/Umar/brain_experiments/eval_data')

    args = parser.parse_args()
    visualize(args.model, args.data_root)