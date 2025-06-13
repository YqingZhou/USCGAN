import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train1"
VAL_DIR = "data/val1"
BATCH_SIZE = 1
LEARNING_RATE = 0.000025
LEARNING_RATE1 = 0.00002
LAMBDA_IDENTITY = 1
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 151
NUM_EPOCHS_1 = 101
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "horse2zebra.pth"
CHECKPOINT_GEN_Z = "zebra2horse.pth"
# CHECKPOINT_CRITIC_H = "critich.pth.tar"
# CHECKPOINT_CRITIC_Z = "criticz.pth.tar"


transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

