import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image
import argparse

from Prepare import get_dataloader, get_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--tl', default=1, type=int, metavar='N',
                        help='flag pre training/transfer learning')
    args = parser.parse_args()

    # Задание путей к изображениям для обучения
    BASE_PATH = "crops_greyscale"

    TRAIN = os.path.join(BASE_PATH, "train")
    VAL = os.path.join(BASE_PATH, "test")

    # Число изображений в батче
    FEATURE_EXTRACTION_BATCH_SIZE = 8
    EPOCHS = args.epochs
    LR = 0.005

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    IMAGE_SIZE = 200
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Аугментация изображений
    trainTransform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    valTransform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    (trainDS, trainLoader) = get_dataloader(TRAIN,
        transforms=trainTransform, batchSize=FEATURE_EXTRACTION_BATCH_SIZE)
    (valDS, valLoader) = get_dataloader(VAL,
        transforms=valTransform, batchSize=FEATURE_EXTRACTION_BATCH_SIZE,
        shuffle=False)
    
    model = get_model(args.tl)
    numFeatures = model.fc.in_features

    model.fc = nn.Linear(numFeatures, len(trainDS.classes))
    model = model.to(DEVICE)

    



if __name__ == '__main__':
    main()