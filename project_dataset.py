import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoTokenizer
from project_config import Config, TARGET_COLUMN, TEXT_COLUMN, ID_COLUMN, AUX_COLUMN
import timm
import torch


class MultimodalDataset(Dataset):

    def __init__(self, config: Config, transforms: A.Compose, df: pd.DataFrame, mask = None):
        self.df = df.reset_index(drop=True)
        self.image_cfg = timm.get_pretrained_cfg(config.image_model_name)
        self.transforms = transforms
        self.mask = mask
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, TEXT_COLUMN]
        target = self.df.loc[idx, TARGET_COLUMN]
        mass = self.df.loc[idx, AUX_COLUMN]
        img_path = os.path.join(self.config.data_path, "images", self.df.loc[idx, ID_COLUMN], "rgb.png")
        image = Image.open(img_path).convert('RGB')
        if self.mask == 'image':
            text = 'text'
        elif self.mask == 'text':
            image = np.zeros((480, 640, 3), dtype=np.uint8)

        image = self.transforms(image=np.array(image))["image"]

        return {"target": target, "image": image, "text": text, "aux": mass}

def worker_init_fn(worker_id):
    # Каждый воркер DataLoader получает копию одного и того же генератора
    # albumentations, поэтому без пересева аугментации совпадают между
    # воркерами и повторяются каждую эпоху. torch.initial_seed() уникален
    # для воркера и эпохи и воспроизводим при torch.manual_seed(config.seed).
    dataset = torch.utils.data.get_worker_info().dataset
    dataset.transforms.set_random_seed(torch.initial_seed() % 2**32)

def collate_fn(batch, tokenizer):
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    targets = torch.tensor([item["target"] for item in batch], dtype=torch.float32)
    auxillarities = torch.tensor([item["aux"] for item in batch], dtype=torch.float32).unsqueeze(1)

    tokenized_input = tokenizer(texts,
                                return_tensors="pt",
                                padding=True,
                                truncation=True)

    return {
        "target": targets,
        "image": images,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"],
        "aux": auxillarities
    }


def get_transforms(config: Config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.image_model_name)
    height, width = cfg.input_size[1], cfg.input_size[2]

    resize = A.Resize(height=height, width=width)

    if ds_type == "train":
        transforms = A.Compose(
            [
                resize,
                #съемка сверху: у блюда нет "верха", повороты и отражения
                #не меняют калорийность
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                #умеренный масштаб
                A.Affine(scale=(0.9, 1.1),
                         rotate=(-20, 20),
                         translate_percent=(-0.05, 0.05),
                         fill=0,
                         p=0.7),
                #освещение меняем слабо, оттенок почти не трогаем
                #цвет говорит о продукте и способе приготовления
                A.ColorJitter(brightness=0.15,
                              contrast=0.15,
                              saturation=0.1,
                              hue=0.02,
                              p=0.5),
                # мелкие дырки, чтобы не спрятать заметную часть еды
                A.CoarseDropout(num_holes_range=(1, 4),
                                hole_height_range=(0.05, 0.1),
                                hole_width_range=(0.05, 0.1),
                                fill=0,
                                p=0.3),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                ToTensorV2(p=1.0)
            ],
            seed=config.seed,
        )
    else:
        transforms = A.Compose(
            [
                resize,
                A.Normalize(mean=cfg.mean, std=cfg.std),
                ToTensorV2(p=1.0)
            ]
        )

    return transforms