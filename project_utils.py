import os
import random
from functools import partial

import numpy as np
import pandas as pd

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torchmetrics

from transformers import AutoModel, AutoTokenizer

from project_dataset import MultimodalDataset, collate_fn, get_transforms, worker_init_fn

from project_config import Config, DEVICE

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def set_requires_grad(module: nn.Module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        hidden_dim = config.hidden_dim
        self.text_model = AutoModel.from_pretrained(config.text_model_name)
        self.image_model = timm.create_model(
            config.image_model_name,
            pretrained=True,
            num_classes=0
        )

        self.text_proj = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.image_proj = nn.Sequential(nn.Linear(self.image_model.num_features, config.hidden_dim),
                                        nn.LayerNorm(hidden_dim))

        # массу учитываем в виде небольшогр эмбеддинга
        self.mass_proj = nn.Linear(1, hidden_dim // 4)


        self.head = nn.Sequential(
            nn.Linear(2*hidden_dim + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, input_ids, attention_mask, image, aux):
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:,  0, :]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)
        mass_emb = self.mass_proj((aux - 5.10)/0.82) # нормализация массы, иначе будет доминировать над остальными признаками

        fused_emb = torch.cat([text_emb, image_emb, mass_emb], dim=1)

        return self.head(fused_emb).squeeze(1)


def train(config: Config, df_train: pd.DataFrame, df_val: pd.DataFrame, mask = None):
    seed_everything(config.seed)

    # Инициализация модели
    model = MultimodalModel(config).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)

    set_requires_grad(model.text_model,
                      unfreeze_pattern=config.text_model_unfreeze, verbose=True)
    set_requires_grad(model.image_model,
                      unfreeze_pattern=config.image_model_unfreeze, verbose=True)

    # Оптимизатор с разными LR
    optimizer = AdamW([{
        'params': model.text_model.parameters(),
        'lr': config.text_lr
    }, {
        'params': model.image_model.parameters(),
        'lr': config.image_lr
    }, {'params': [*model.text_proj.parameters(), *model.image_proj.parameters(),
                *model.mass_proj.parameters(), *model.head.parameters()], 'lr': config.head_lr}
        ], weight_decay=1e-2)

    criterion = nn.SmoothL1Loss() # Huber в лог-пространстве

    # Загрузка данных
    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type="val")
    train_dataset = MultimodalDataset(config, transforms, df_train, mask)
    val_dataset = MultimodalDataset(config, val_transforms, df_val, mask)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              collate_fn=partial(collate_fn,
                                                 tokenizer=tokenizer),
                              num_workers=4, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            collate_fn=partial(collate_fn,
                                               tokenizer=tokenizer),
                            num_workers=4, worker_init_fn=worker_init_fn)

    best_mae_val = float('inf')
    history = []

    print("training started")
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        train_abs_err, n = 0.0, 0

        for batch in train_loader:
            # Подготовка данных
            inputs = {
                'input_ids': batch['input_ids'].to(DEVICE),
                'attention_mask': batch['attention_mask'].to(DEVICE),
                'image': batch['image'].to(DEVICE),
                'aux': batch['aux'].to(DEVICE),
                }
            target = batch['target'].to(DEVICE)

            # Forward
            optimizer.zero_grad()
            pred_log = model(**inputs)
            loss = criterion(pred_log, target)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # MAE в ккал: из лог-шкалы обратно через expm1
            pred = torch.expm1(pred_log.detach()).clamp(min=0)
            train_abs_err += (pred - torch.expm1(target)).abs().sum().item()
            n += target.size(0)


        # Валидация
        train_loss = total_loss / len(train_loader)
        train_mae = train_abs_err / n
        val_loss, val_mae = validate(model, val_loader, DEVICE, criterion)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_mae,
            "val_mae": val_mae,
        })

        print(
            f"Epoch {epoch}/{config.epochs-1} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} "
            f"| Train MAE: {train_mae:.1f} | Val MAE: {val_mae:.1f}"
        )

        if val_mae < best_mae_val:
            print(f"New best model, epoch: {epoch}")
            best_mae_val = val_mae
            torch.save(model.state_dict(), config.save_path)

    return  pd.DataFrame(history)


def validate(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    abs_err, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: batch[k].to(device)
                      for k in ('input_ids', 'attention_mask', 'image', 'aux')}
            target = batch['target'].to(device)
            pred_log = model(**inputs)

            total_loss += criterion(pred_log, target).item()
            pred = torch.expm1(pred_log).clamp(min=0)
            abs_err += (pred - torch.expm1(target)).abs().sum().item()
            n += target.size(0)
    return total_loss / len(val_loader), abs_err / n
