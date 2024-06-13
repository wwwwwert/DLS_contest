from os import path, walk
from pathlib import Path
from typing import Any, Dict, List, Tuple

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from lightning_fabric.utilities.seed import seed_everything
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm
from transformers import (BertForSequenceClassification, BertTokenizer,
                          Trainer, TrainingArguments)
import json

seed_everything(10)


def classify(
        model_filename: str,
        test_img_dir: str
    ):
    filenames = []
    for (dirpath, dirnames, files) in walk(test_img_dir):
        filenames.extend(files)
    decoy_gt = {filename: 0 for filename in filenames}
    imgs_dataset = ImgClassDataset(
        decoy_gt, 
        test_img_dir, 
        (160, 160),
        train=False,
        fast_train=False
    )
    # imgs_dataset.to(torch.device('cuda'))
    model = ModelLightning.load_from_checkpoint(model_filename)
    model.to(torch.device('cuda'))
    model.eval()
    preds = {}
    for i, filename in tqdm(enumerate(filenames), total=len(filenames)):
        img, _ = imgs_dataset[i]
        img.to(torch.device('cuda'))
        pred = model(img.unsqueeze(0).to(torch.device('cuda'))).cpu().detach().numpy()
        preds[filename] = pred.argmax()
    return preds


class MyDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            split='train'
        ) -> None:
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
        tokenized_save = Path(f'tokenized_{split}.json')
        if tokenized_save.exists():
            print('loading  tokenized...')
            with open(tokenized_save, 'r') as fp:
                self.tokenized = json.load(fp)
        else:
            print('tokenizing dataset...')
            texts = (self.data['Category'] + ' ' + self.data['Subtitles']).to_list()
            self.tokenized = self.tokenizer(texts, padding=True, truncation=True, max_length=512)
            with open(tokenized_save, 'w') as fp:
                json.dump(self.tokenized.data, fp, indent=2)
        self.labels = self.data['ViewCount'].astype('float32').to_list()
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        ids = torch.tensor(self.tokenized['input_ids'][index])
        mask = torch.tensor(self.tokenized['attention_mask'][index])
        target = torch.tensor(self.labels[index])
        return ids, mask, target
    

class FineTuneModel(nn.Module):
    def __init__(self, freeze: bool=True) -> None:
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased")
        if freeze:
            # for param in self.bert.bert.parameters():
            #     param.requires_grad = False
            for child in list(self.bert.bert.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False
        self.dense = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = output['hidden_states'][-1]
        # print(output.keys())
        # print(last_hidden_state.shape)
        embedding = last_hidden_state.sum(axis=1)
        # print(embedding.shape)
        y = self.dense(embedding)
        # print(y.shape)
        return y.squeeze()


class ModelLightning(pl.LightningModule):
    def __init__(self, freeze: bool=False) -> None:
        super().__init__()
        self.model = FineTuneModel(freeze)
        self.criterion = torch.nn.MSELoss()
        self.mae = torchmetrics.regression.MeanAbsoluteError()

    def configure_optimizers(self):
        """
        Construct optimizer. Training this strange model with vanilla stochastic
        gradient descent is tough, so we use momentum
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
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
            "monitor": "val_mae",
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
        x_batch, mask, y_batch = train_batch
        y_pred = self.model(**{
            'input_ids': x_batch,
            'attention_mask': mask
        })
        loss = self.criterion(y_pred, y_batch)

        mae = self.mae(y_pred, y_batch)

        metrics = {"train_mae": mae, "train_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, logger=True)


        return loss
    
    def validation_step(self, batch, batch_idx):
        x_batch, mask, y_batch = batch
        y_pred = self.model(**{
            'input_ids': x_batch,
            'attention_mask': mask
        })
        loss = self.criterion(y_pred, y_batch)

        mae = self.mae(y_pred, y_batch)

        metrics = {"val_mae": mae, "val_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return metrics
    
    def forward(self, imgs) -> Any:
        return self.model(imgs)