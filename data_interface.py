from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, RandomCrop, ToPILImage, GaussianBlur, Normalize
import numpy as np
import torch
import cv2 as cv
import torchvision.transforms.functional as TF
import random

class DriveDataset(Dataset):
    def __init__(self, data_dir, split, transforms, ) -> None:
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.transforms = transforms

        self.all_imgs = os.listdir(os.path.join(data_dir, split, 'raw'))
        self.all_gt = os.listdir(os.path.join(data_dir, split, 'gt'))
        self.all_masks = os.listdir(os.path.join(data_dir, split, 'mask'))
    
    def __len__(self):
        return len(self.all_imgs)
    
    def __getitem__(self, index):
        img_name = self.all_imgs[index]
        gt_name = img_name.split('_')[0] + '_' + 'manual1.gif'
        mask_name = img_name.split('_')[0] + '_' + 'training_mask.gif'
        img_path = os.path.join(self.data_dir, self.split, 'raw', img_name)
        gt_path = os.path.join(self.data_dir, self.split, 'gt', gt_name)
        mask_path = os.path.join(self.data_dir, self.split, 'mask', mask_name)
        assert img_name.split('_')[0] == gt_name.split('_')[0]
        
        # Open Image
        img = Image.open(img_path)
        gt = Image.open(gt_path)
        mask = Image.open(mask_path)
        if self.transforms:
            img_transformed, gt_transformed = self.transforms(img, gt, mask)

        return img_transformed, gt_transformed

class CustomTransform:
    def __init__(self, angle_range=(-10, 11), img_size: int = 448, crop_size: int = 224) -> None:
        self.angle = angle_range
        self.img_size = img_size
        self.crop_size = crop_size

    def CLAHE(self, img: torch.Tensor) -> torch.Tensor:
        img = ToPILImage()(img)
        img = cv.cvtColor(np.array(img), cv.COLOR_RGB2LAB)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img[:,:,0] = clahe.apply(img[:,:,0])
        img = cv.cvtColor(img, cv.COLOR_LAB2RGB)
        return ToTensor()(img)

    # fill black with avg of the image all RGB channels
    def fill_black_with_avg(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        avg_r = torch.mean(img[0, :, :][mask.squeeze(0) == 1])
        avg_b = torch.mean(img[1, :, :][mask.squeeze(0) == 1])
        avg_g = torch.mean(img[2, :, :][mask.squeeze(0) == 1])
        avg = torch.tensor([avg_r, avg_b, avg_g])
        # fill all black pixels with avg
        img[:, mask.squeeze(0) == 0] = avg.unsqueeze(1)

        return img
        
    def transform(self, img, gt, mask):
        img = Resize((self.img_size, self.img_size))(ToTensor()(img))
        gt = Resize((self.img_size, self.img_size))(ToTensor()(gt))
        mask = Resize((self.img_size, self.img_size))(ToTensor()(mask))
        img = self.CLAHE(img)
        self.rotation_angle = random.randrange(*self.angle)
        img = TF.rotate(img, self.rotation_angle, expand=False)
        gt = TF.rotate(gt, self.rotation_angle, expand=False)
        mask = TF.rotate(mask, self.rotation_angle, expand=False)
        # Fill black with avg value
        # img = self.fill_black_with_avg(img, mask)
        
        # h_flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            gt = TF.hflip(gt)
        if random.random() > 0.5:
            img = TF.vflip(img)
            gt = TF.vflip(gt)

        if random.random() > 0.5:
            img = GaussianBlur(3, sigma=(0.1, 2.0))(img)
        # if random.random() > 0.5:
        #     img = TF.adjust_brightness(img, random.random()+0.5)
        # if random.random() > 0.5:
        #     img = TF.adjust_contrast(img, random.random()+0.5)
        
        if self.crop_size != self.img_size:
            cropper = RandomCrop((self.crop_size, self.crop_size))
            left, right, h, w = cropper.get_params(img, (self.crop_size, self.crop_size))
            img = Resize((self.crop_size, self.crop_size))(TF.crop(img, left, right, h, w))
            gt = Resize((self.crop_size, self.crop_size))(TF.crop(gt, left, right, h, w))
        return img, gt
    
    def __call__(self, img, gt, mask):
        return self.transform(img, gt, mask)

class DriveDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = "data",
                 batch_size: int = 32,
                 numworkers: int = 1,
                 img_size: int = 448,
                 transform_angle: tuple = (-10, 11),
                 crop_size: int = 224) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.numworkers = numworkers
        self.img_size = img_size
        self.transform_angle = transform_angle
        self.crop_size = crop_size
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = DriveDataset(split="train",
                                              transforms=CustomTransform(angle_range=self.transform_angle,
                                                                         img_size=self.img_size,
                                                                         crop_size=self.crop_size),
                                              data_dir=self.data_dir)
            self.val_dataset = DriveDataset(split="val",
                                            transforms=CustomTransform(angle_range=self.transform_angle,
                                                                       img_size=self.img_size,
                                                                       crop_size=self.crop_size),
                                            data_dir=self.data_dir)
        if stage == "test" or stage is None:
            self.test_dataset = DriveDataset(split="test",
                                             transforms=CustomTransform(angle_range=self.transform_angle,
                                                                        img_size=self.img_size,
                                                                        crop_size=self.crop_size),
                                             data_dir=self.data_dir)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.numworkers
                          )
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.numworkers
                          )
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.numworkers
                          )  
        
        