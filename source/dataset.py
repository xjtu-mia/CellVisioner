import os
import torch.utils.data as data

import random
import numpy as np
import cv2
import torch



import albumentations as albu

def load_augmentations(self):
    tfrd = albu.Compose([
        # albu.RandomCrop(height=512,width=512,always_apply=True,p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
    ])
    return tfrd


class muti_channel_Dataset(data.Dataset):
    # get paired dataset
    def __init__(self, dataroot, train_type, opts, augment):
        self.dataroot = dataroot
        self.normalize = opts.is_norm
        self.augment = augment
        self.inch = opts.inch
        if self.augment:
            self.augmentations = load_augmentations(self)
        self.A_path_i, self.A_path_gt = os.path.join(self.dataroot, train_type, "bright"), \
                                        os.path.join(self.dataroot, train_type,  f'{opts.flu}_add')
        
        self.listA = os.listdir(self.A_path_i)
        self.listA.sort(key=lambda x: int(x[:-4]))
        self.dataset_size = len(self.listA)
    
    def __getitem__(self, index):
        A_gt, = cv2.imread(os.path.join(self.A_path_gt, self.listA[index * self.inch]), 0),
        
        A = np.zeros((A_gt.shape[0], A_gt.shape[1], self.inch), dtype=A_gt.dtype)
        for i, name in enumerate(self.listA[index * self.inch:(index + 1) * self.inch]):
            img = cv2.imread(os.path.join(self.A_path_i, name), 0)
            A[:, :, i] = img
        
        if self.normalize:
            A, A_gt = A / 255, A_gt / 255,
        
        else:
            A, A_gt = (A - np.mean(A)) / np.std(A), (A_gt - np.mean(A_gt)) / np.std(A_gt)
        
        if self.augment:
            augmented = self.augmentations(image=A, mask=A_gt)
            A = augmented["image"]
            A_gt = augmented["mask"]
        
        A = torch.from_numpy(A.copy().transpose(2, 0, 1)).float()
        A_gt = torch.from_numpy(A_gt.copy()).unsqueeze(0).float()  #
        
        return A, A_gt, self.listA[index * self.inch]
    
    def __len__(self):
        return self.dataset_size // self.inch


class mask_muti_channel_Dataset(data.Dataset):
    # get paired dataset
    def __init__(self, dataroot, train_type, opts, augment):
        self.dataroot = dataroot
        self.normalize = opts.is_norm
        self.augment = augment
        self.inch = opts.inch
        if self.augment:
            self.augmentations = load_augmentations(self)
        self.A_path_i, self.A_path_gt = os.path.join(self.dataroot, train_type, "bright"), \
                                        os.path.join(self.dataroot, train_type, f'{opts.flu}_add')
        
        self.listA = os.listdir(self.A_path_i)
        self.listA.sort(key=lambda x: int(x[:-4]))
        
        self.dataset_size = len(self.listA)
    
    def __getitem__(self, index):
        A_gt, = cv2.imread(os.path.join(self.A_path_gt, self.listA[index * self.inch]), 0),
        
        A = np.zeros((A_gt.shape[0], A_gt.shape[1], self.inch), dtype=A_gt.dtype)
        for i, name in enumerate(self.listA[index * self.inch:(index + 1) * self.inch]):
            img = cv2.imread(os.path.join(self.A_path_i, name), 0)
            A[:, :, i] = img
        mask = A_gt
        mask = np.where(mask > 5, 1, 0)
        if self.normalize:
            A, A_gt = A / 255, A_gt / 255,
        
        else:
            A, A_gt = (A - np.mean(A)) / np.std(A), (A_gt - np.mean(A_gt)) / np.std(A_gt)
        masks = np.zeros((A_gt.shape[0], A_gt.shape[1], 2), dtype=A_gt.dtype)
        masks[:, :, 0] = A_gt
        masks[:, :, 1] = mask
        if self.augment:
            augmented = self.augmentations(image=A, mask=masks)
            A = augmented["image"]
            A_gt = augmented["mask"][:, :, 0]
            mask = augmented["mask"][:, :, 1]
        
        A = torch.from_numpy(A.copy().transpose(2, 0, 1)).float()
        A_gt = torch.from_numpy(A_gt.copy()).unsqueeze(0).float()  ###
        mask = torch.from_numpy(mask.copy()).unsqueeze(0).float()
        return A, A_gt, mask, self.listA[index * self.inch]
    def __len__(self):
        return self.dataset_size // self.inch
    
    
    
class testDataset(data.Dataset):
    def __init__(self, dataroot, opts, augment):
        self.dataroot = dataroot
        self.normalize = opts.is_norm
        self.augment = augment
        if self.augment:
            self.augmentations = load_augmentations(self)
        
        # self.standardize = True
        self.A_path_i = self.dataroot
        
        self.listA = os.listdir(self.A_path_i)
        self.listA.sort(key=lambda x: int(x[:-4]))
        self.dataset_size = len(self.listA)
    
    def __getitem__(self, index):
        A = cv2.imread(os.path.join(self.A_path_i, self.listA[index]), 0)
        if self.normalize:
            A, A_gt = A / 255
        else:
            A = (A - np.mean(A)) / np.std(A)
        if self.augment:
            augmented = self.augmentations(image=A)
            A = augmented["image"]
        A = torch.from_numpy(A.copy()).unsqueeze(0).float()  ###
        
        return A, self.listA[index]
    
    def __len__(self):
        return self.dataset_size



