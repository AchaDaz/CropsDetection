import os
import numpy as np
import shutil
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models


def get_dataloader(device, rootDir, transforms, batchSize, shuffle=True):
    '''
        Функция для создания дата лоадера из директорий с изображениями
    '''
    ds = datasets.ImageFolder(root=rootDir,
        transform=transforms)
    loader = DataLoader(ds, batch_size=batchSize,
        shuffle=shuffle,
        num_workers=os.cpu_count(),
        pin_memory=True if device == "cuda" else False)
    return (ds, loader)

def get_model(tl: int):
    if tl == 0:
        model = models.resnet34(pretrained=False)
    else:
        model = torch.load('results/experiment2/nn_pretrained/nn27')
        for param in model.parameters():
            param.requires_grad = False

    return model

def train_test_split():
    '''
        Функция для разделения датасета на тренировочную и тестовую выборки
    '''
    print("########### Train Test Val Script started ###########")
    #data_csv = pd.read_csv("DataSet_Final.csv") ##Use if you have classes saved in any .csv file

    root_dir = 'dataset'
    classes_dir = ['aluminium_foil',
                   'brown_bread',
                   'corduroy',
                   'cotton',
                   'cracker',
                   'linen',
                   'orange_peel',
                   'sandpaper',
                   'sponge',
                   'styrofoam']

    #for name in data_csv['names'].unique()[:10]:
    #    classes_dir.append(name)

    processed_dir = 'KTH_TIPS'

    val_ratio = 0.10
    test_ratio = 0.05

    for cls in classes_dir:
        # Creating partitions of the data after shuffeling
        print("$$$$$$$ Class Name " + cls + " $$$$$$$")
        src = processed_dir +"//" + cls  # Folder to copy images from

        allFileNames = os.listdir(src)
        np.random.shuffle(allFileNames)
        train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                                  [int(len(allFileNames) * (1 - (val_ratio + test_ratio))),
                                                                   int(len(allFileNames) * (1 - val_ratio)),
                                                                   ])

        train_FileNames = [src + '//' + name for name in train_FileNames.tolist()]
        val_FileNames = [src + '//' + name for name in val_FileNames.tolist()]
        test_FileNames = [src + '//' + name for name in test_FileNames.tolist()]

        print('Total images: '+ str(len(allFileNames)))
        print('Training: '+ str(len(train_FileNames)))
        print('Validation: '+  str(len(val_FileNames)))
        print('Testing: '+ str(len(test_FileNames)))

        # # Creating Train / Val / Test folders (One time use)
        os.makedirs(root_dir + '/train//' + cls)
        os.makedirs(root_dir + '/val//' + cls)
        os.makedirs(root_dir + '/test//' + cls)

        # Copy-pasting images
        for name in train_FileNames:
            shutil.copy(name, root_dir + '/train//' + cls)

        for name in val_FileNames:
            shutil.copy(name, root_dir + '/val//' + cls)

        for name in test_FileNames:
            shutil.copy(name, root_dir + '/test//' + cls)

    print("########### Train Test Val Script Ended ###########")