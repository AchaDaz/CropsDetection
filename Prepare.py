import os
from torch.utils.data import DataLoader
from torchvision import datasets

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