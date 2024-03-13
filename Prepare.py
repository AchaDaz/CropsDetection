import os
import random
import numpy as np
import shutil
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn.functional import softmax
from PIL import Image, ImageDraw

def setup():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    dist.init_process_group('gloo')

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def evaluate(model, device, test_loader):

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


def cleanup():
    dist.destroy_process_group()

def get_dataloader(device, rootDir, transforms, batchSize):
    '''
        Функция для создания дата лоадера из директорий с изображениями
    '''
    ds = datasets.ImageFolder(root=rootDir,
                              transform=transforms)
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    loader = DataLoader(ds, 
                        batch_size=batchSize,
                        num_workers=os.cpu_count(),
                        pin_memory=True if device == "cuda" else False,
                        sampler=sampler)
    return (ds, loader, sampler)

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

def base_check(model):
    device = 'cpu'
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder('crops/train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    classes = train_dataset.classes

    # Визуализация предсказаний
    sample_image = Image.open('./crops_greyscale/test/cabbage/Screenshot_1.jpg')
    sample_image_tensor = transform(sample_image).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        output = model(sample_image_tensor)
        _, predicted_index = torch.max(output.data, 1)
        prob = torch.nn.functional.softmax(output, dim=1)
    
    if prob.max().item() > 0.7:
        predicted_class = classes[predicted_index.item()]
    else:
        predicted_class = 'None'

    plt.imshow(sample_image)
    plt.axis('off')
    plt.title(f'Predicted Class: {predicted_class}')
    plt.show()

def crop_predict_image(path_image: str, model):
    '''
    Функция принимает на вход путь к изображению, разрезает его,
    каждый фрагмент отправляет в модель.
    Выходное значение - список тензоров с вероятноястями.
    '''
        
    device = torch.device("cpu")
        
    image = Image.open(path_image)
    width, height = image.size
    
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    
    train_dataset = datasets.ImageFolder('crops/train', transform=transform)
    classes = train_dataset.classes
    
    # Список из списков с тензорами вероятностей классов
    predicted_output = [[] for lst in range(0, 10)]
    
    # Настройки для отрисовывания
    colors = ['red', 'blue', 'yellow', 'green', 'orange', 'brown', 'white', 'black']
    
    classes_colors = dict(zip(range(0, 100), zip(classes, colors)))
    
    draw = ImageDraw.Draw(image)
    
    upper = 0
    lower = round(height * 0.1)
    
    row = -1
    
    for h in range(0, 10):
        row += 1
        left = 0
        right = round(width * 0.1)
        
        for w in range(0, 10):
            cropped_image = image.crop((left, upper, right, lower))
            transfromed_image = transform(cropped_image).unsqueeze(0).to(device)
            
            predicted = softmax(model(transfromed_image), -1)
            
            predicted_output[row].append(predicted)
            
            if float(torch.max(predicted)) > 0.9:
                index = int(torch.argmax(predicted))
                
                draw.line((left, upper, right, upper), fill=classes_colors[index][1], width=10)
                draw.line((right, upper, right, lower), fill=classes_colors[index][1], width=10)
                draw.line((left, lower, right, lower), fill=classes_colors[index][1], width=10)
                draw.line((left, lower, left, upper), fill=classes_colors[index][1], width=10)
            
            left += round(width * 0.1)
            right += round(width * 0.1)
            
        upper += round(height * 0.1)
        lower += round(height * 0.1)
    
    text = ''
    for i in classes_colors:
        text = text + classes_colors[i][0] + ' - ' + classes_colors[i][1] + ' \n'
    
    plt.figure(figsize=(16, 12))
    plt.imshow(image)
    plt.axis('off')
    plt.title(text)
    plt.show()
    
    return image, predicted_output, classes