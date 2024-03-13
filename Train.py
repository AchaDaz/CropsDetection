import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.multiprocessing as mp
import torch.distributed as dist
from datetime import datetime
import time
from tqdm import tqdm

from prepare import get_dataloader, get_model

# Задание путей к изображениям для обучения
BASE_PATH = "crops_greyscale"

TRAIN = os.path.join(BASE_PATH, "train")
VAL = os.path.join(BASE_PATH, "test")

# Число изображений в батче
FEATURE_EXTRACTION_BATCH_SIZE = 8
LR = 0.005

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train(tl, epochs, local_rank):
    torch.cuda.set_per_process_memory_fraction(0.2, device=0)
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

    #rank = args.nr * args.gpus + gpu
    #dist.init_process_group(backend='gloo', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)

    (trainDS, trainLoader, trainSampler) = get_dataloader(DEVICE,
                                            TRAIN,
                                            transforms=trainTransform,
                                            batchSize=FEATURE_EXTRACTION_BATCH_SIZE,
                                            )
    
    (valDS, valLoader, valSampler) = get_dataloader(DEVICE,
                                        VAL,
                                        transforms=valTransform,
                                        batchSize=FEATURE_EXTRACTION_BATCH_SIZE,
                                        )

    model = get_model(tl)
    numFeatures = model.fc.in_features

    model.fc = nn.Linear(numFeatures, len(trainDS.classes))
    model = model.to(DEVICE)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    lossFunc = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDS) // FEATURE_EXTRACTION_BATCH_SIZE
    valSteps = len(valDS) // FEATURE_EXTRACTION_BATCH_SIZE

    # loop over epochs
    print("[INFO] training the network...")
    # initialize a dictionary to store training history
    H = {"train_loss": [], "train_acc": [], "val_loss": [],
        "val_acc": []}

    startTime = time.time()
    for e in tqdm(range(epochs)):
        # set the model in training macode
        model.train()
        trainSampler.set_epoch(e)
        valSampler.set_epoch(e)
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0
        # loop over the training set
        for (i, (x, y)) in enumerate(trainLoader):
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            # perform a forward pass and calculate the training loss
            pred = model(x)
            loss = lossFunc(pred, y)
            # calculate the gradients
            loss.backward()
            # check if we are updating the model parameters and if so
            # update them, and zero out the previously accumulated gradients
            if (i + 2) % 2 == 0:
                opt.step()
                opt.zero_grad()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()
            
            
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (x, y) in valLoader:
                # send the input to the device
                (x, y) = (x.to(DEVICE), y.to(DEVICE))
                # make the predictions and calculate the validation loss
                pred = model(x)
                totalValLoss += lossFunc(pred, y)
                # calculate the number of correct predictions
                valCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDS)
        valCorrect = valCorrect / len(valDS)
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(
            avgValLoss, valCorrect))
        
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))
    date = datetime.now()
    save_dir = f'results/experiment2/{date.date()}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if local_rank == 0 and tl == 1:
        torch.save(model, f'{save_dir}/tl_{date.time().strftime("%H.%M.%S")}.pt')
    if local_rank == 0 and tl == 0:
        torch.save(model, f'{save_dir}/pre_{date.time().strftime("%H.%M.%S")}.pt')
    
    return model, H
