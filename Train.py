import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image

from Prepare import get_dataloader

# Задание путей к изображениям для обучения
BASE_PATH = "crops_greyscale"

TRAIN = os.path.join(BASE_PATH, "train")
VAL = os.path.join(BASE_PATH, "test")

# Число изображений в батче
FEATURE_EXTRACTION_BATCH_SIZE = 8
EPOCHS = 10
LR = 0.005

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



