from os import path, walk
from pathlib import PosixPath
from typing import Any, Dict, List, Tuple

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from lightning_fabric.utilities.seed import seed_everything
from model.UNet import UNet
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

seed_everything(10)

def restore(
        model: str,
        files: str
    ):
    transforms = A.Compose([
                A.Normalize(),
                # A.Resize(160, 160)
        ])
    
    preds = []
    with torch.no_grad():
        model.to(torch.device('cuda'))
        for img in tqdm(files):
            img = transforms(image=img)['image']
            img = torch.from_numpy(img.astype('float32')).permute(2, 0, 1).contiguous()
            img.to(torch.device('cuda'))
            pred = model(img.unsqueeze(0).to(torch.device('cuda'))).cpu().detach().squeeze().permute(1, 2, 0).numpy().reshape([128, 128, 3])
            img = img.permute(1, 2, 0).cpu().detach().numpy()
            restored = img - pred
            restored = denormalize(restored)
            preds.append(restored)
    return np.array(preds)

def denormalize(img):
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    max_pixel_value = 255
    img = img * stds * max_pixel_value + means * max_pixel_value
    return img.clip(0, 255).astype('uint8')

class ImgClassDataset(Dataset):
    def __init__(
            self,
            true_images: np.array,
            corrupted_images: np.array,
            train=True
        ) -> None:
        super().__init__()
        self.true_images = true_images
        self.corrupted_images = corrupted_images

        self.train_transforms = A.Compose([
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                # A.RandomBrightnessContrast(p=0.5),
                # A.Equalize(p=0.7),

                A.Normalize(),
                # A.Resize(160, 160),
        ])

        self.test_transforms = A.Compose([
                A.Normalize(),
                # A.Resize(160, 160)
        ])
        self.train=train

    def __len__(self):
        return self.true_images.shape[0]

    def __getitem__(self, index) -> Any:
        corrupted = self.corrupted_images[index].astype('uint8')
        corrupted = self.augment_img(corrupted)
        corrupted = torch.from_numpy(corrupted.astype('float32')).permute(2, 0, 1).contiguous()

        true = self.true_images[index].astype('uint8')
        true = self.augment_img(true)
        true = torch.from_numpy(true.astype('float32')).permute(2, 0, 1).contiguous()

        mask = corrupted - true
        
        return corrupted, true, mask
    

    def augment_img(self, img: np.array):
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
            
        if self.train:
            transformed = self.train_transforms(image=img)
        else:
            transformed = self.test_transforms(image=img)
        transformed_image = transformed['image']

        return transformed_image
    
class ModelLightning(pl.LightningModule):
    def __init__(self, freeze: str='full') -> None:
        super().__init__()
        self.model = UNet(3, 3)
        self.criterion = torch.nn.MSELoss()
        self.mae = torchmetrics.regression.MeanAbsoluteError()

    def configure_optimizers(self):
        """
        Construct optimizer. Training this strange model with vanilla stochastic
        gradient descent is tough, so we use momentum
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=2,
            verbose=True,
        )
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "train_mae",
        }

        return [optimizer], [lr_dict]

    def training_step(self, train_batch, batch_idx):
        """
        Define training step:
        1. pass batch samples into the model
        2. get batch predictions from the model
        3. compute loss using batch labels and predictions
        4. log loss to pytorch lightning logger
        """
        x_batch, y_batch, mask_batch = train_batch
        y_pred = self.model(x_batch)
        y_pred = y_pred.reshape([-1, 3 * 128 * 128])
        y_batch = y_batch.reshape([-1, 3 * 128 * 128])
        mask_batch = mask_batch.reshape([-1, 3 * 128 * 128])
        loss = self.criterion(y_pred, mask_batch)

        mae = self.mae(y_pred, mask_batch)

        metrics = {"train_mae": mae, "train_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, logger=True)


        return loss
    
    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_pred = self.model(x_batch).reshape([-1, 49152])
        loss = self.criterion(y_pred, y_batch)

        pred_class = nn.functional.softmax(y_pred, dim=1).argmax(dim=1)
        mae = self.mae(pred_class, y_batch)

        metrics = {"val_mae": mae, "val_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return metrics
    
    def forward(self, imgs) -> Any:
        return self.model(imgs)