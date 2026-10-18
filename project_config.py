import torch
from dataclasses import dataclass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERT_MODEL = "bert-base-uncased"
IMAGE_MODEL = "tf_efficientnet_b0"
DATA_PATH = 'data\\nutrition'
SAVE_PATH = "best_dish_model.pth"
TEXT_COLUMN = 'ingredients'
TARGET_COLUMN = 'total_calories'
ID_COLUMN = 'dish_id'
AUX_COLUMN = 'total_mass'

@dataclass
class Config:
    # для воспроизводимости
    seed:int = 42

    # Модели
    text_model_name: str = BERT_MODEL
    image_model_name: str = IMAGE_MODEL

    # Какие слои размораживаем - совпадают с неймингом в моделях
    text_model_unfreeze: str = "encoder.layer.11"
    image_model_unfreeze: str = "blocks.6|conv_head|bn2"

    # Гиперпараметры
    batch_size:int = 16
    text_lr: float = 3e-5
    image_lr: float = 1e-4
    head_lr:float = 1e-3
    epochs:int = 30
    dropout:float = 0.3
    hidden_dim: int = 256

    # Пути
    data_path:str = DATA_PATH
    save_path:str = SAVE_PATH
