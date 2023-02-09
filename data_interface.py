from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
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
    
    def __len__(self):
        return len(self.all_imgs)
    
    def __getitem__(self, index):
        img_name = self.all_imgs[index]
        gt_name = img_name.split('_')[0] + '_' + 'manual1.gif'
        img_path = os.path.join(self.data_dir, self.split, 'raw', img_name)
        gt_path = os.path.join(self.data_dir, self.split, 'gt', gt_name)
        assert img_name.split('_')[0] == gt_name.split('_')[0]
        
        # Open Image
        img = Image.open(img_path)
        gt = Image.open(gt_path)
        if self.transforms:
            img_transformed, gt_transformed = self.transforms(img, gt)

        return img_transformed, gt_transformed

class CustomTransform:
    def __init__(self, angle_range=(-10, 11)) -> None:
        self.angle = angle_range

    def transform(self, img, gt):
        img = Resize((224, 224))(ToTensor()(img))
        gt = Resize((224, 224))(ToTensor()(gt))
        self.rotation_angle = random.randrange(*self.angle)
        img = TF.rotate(img, self.rotation_angle, expand=False)
        gt = TF.rotate(gt, self.rotation_angle, expand=False)
        
        # h_flip
        if random.random() > 0.7:
            img = TF.hflip(img)
            gt = TF.hflip(gt)
        if random.random() > 0.7:
            img = TF.vflip(img)
            gt = TF.vflip(gt)

        return img, gt
    
    def __call__(self, img, gt):
        return self.transform(img, gt)

class DriveDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = "data",
                 batch_size: int = 32,
                 numworkers: int = 1) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.numworkers = numworkers
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = DriveDataset(split="train", transforms=CustomTransform(), data_dir=self.data_dir)
            self.val_dataset = DriveDataset(split="val", transforms=CustomTransform(), data_dir=self.data_dir)
        if stage == "test" or stage is None:
            self.test_dataset = DriveDataset(split="test", transforms=CustomTransform(), data_dir=self.data_dir)
        

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
        
        