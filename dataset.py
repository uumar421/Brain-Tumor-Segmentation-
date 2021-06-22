import torch
from torch.utils import data
import numpy as np
import nibabel as nib

def string_from_num(num):
    num_str = str(num)
    return '0'*(3-len(num_str)) + num_str

class SegmentationDataset(data.Dataset):
    def __init__(self,
                 directory,
                 transform,
                 n_channels=4,
                 n_labels=4,
                 dim=(160,192,128)):
        self.directory=directory
        self.n_channels=n_channels
        self.n_labels=n_labels
        self.dim=dim
        self.transform=transform

    def __len__(self):
        return len(self.directory)

    def __getitem__(self, index):
        x_train = np.zeros((self.n_channels, *self.dim), dtype=np.float64)
        y_train = np.zeros((self.n_labels, *self.dim), dtype=np.float64)
       
        i = index
        file_index, path = self.directory[i]

        image_files = path + '/BraTS20_Training_'+string_from_num(file_index)+'_flair.nii.gz', path +
                       '/BraTS20_Training_'+string_from_num(file_index) +
                       '_t1.nii.gz', path+ '/BraTS20_Training_'+string_from_num(file_index)+
                       '_t1ce.nii.gz', path + '/BraTS20_Training_'+string_from_num(file_index)+'_t2.nii.gz'
        label_file = path + '/BraTS20_Training_'+string_from_num(file_index)+'_seg.nii.gz'

        x_train = np.array([np.array(nib.load(fname).get_fdata()) for fname in image_files])
        y_train = np.array(nib.load(label_file).get_fdata())
        inputs = {'image' : x_train,
                  'label' : y_train}
        inputs = self.transform(inputs)
        return inputs
