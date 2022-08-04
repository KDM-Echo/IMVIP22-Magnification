import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision
import os
from datetime import datetime
import torch.nn.functional as F
import math
now = datetime.now()

ResultsDirectory = rf""# Where results are saved
TopImageDirectory = r"" # Folder where image tiles are kept


from segmentation_models_pytorch.encoders import get_preprocessing_fn

#preprocess_input = get_preprocessing_fn('efficientnet-b2', pretrained='imagenet')

torch.manual_seed(123)

try:
    os.mkdir(ResultsDirectory)
except:
    pass



BatchIDs = []
EpochIDs = []
TrainAccuracies_Batch = []
TrainAccuracies_Epoch = []
TrainLosses_Batch = []
TrainLosses_Epoch = []
TestAccuracies_Batch = []
TestAccuracies_Epoch = []
TestLosses_Batch = []
TestLosses_Epoch = []
Dice_Score_Batch = []
Dice_Score_Epoch = []

############################  Set things up for training #################################################

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm
import torch.optim as optim

# Set up the hyper parameters
LearningRate = 0.0001
Device = "cuda" if torch.cuda.is_available() else "cpu"

BatchSize = 8
Epochs = 5
# Final Image Hieght and width. Image is resized to the longest side first then centre cropped
ImgHeight = 512
ImgWidth = 512
PinMemory = True
LoadModel = True
TestBatches = 7

TrainImageDirectory = TopImageDirectory + r"\Training\Original Images\*\*"
TrainMaskDirectory = TopImageDirectory + r"\Training\Masks\*\*"
TestImageDirectory = TopImageDirectory + r"\Test\Original Images\*\*"
TestMaskDirectory = TopImageDirectory + r"\Test\Masks\*\*"
IndyImageDirectory = TopImageDirectory + r"\Validation\Original Images\*\*"
IndyMaskDirectory = TopImageDirectory + r"\Validation\Masks\*\*"



# Create objects for the model and loss function

#model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1).to(Device)
model = smp.UnetPlusPlus(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=1).to(Device)


#loss_fn = nn.BCEWithLogitsLoss()
optimiser = optim.Adam(model.parameters(), lr=LearningRate)
Scaler = torch.cuda.amp.GradScaler()

# Get lists for the images in each of the above directories

import glob


TestImages = [img for img in glob.glob(TestImageDirectory) if ".png" in img]
TestMasks = [img for img in glob.glob(TestMaskDirectory) if ".png" in img]
TrainImages = [img for img in glob.glob(TrainImageDirectory) if ".png" in img]
TrainMasks = [img for img in glob.glob(TrainMaskDirectory) if ".png" in img]
IndyImages = [img for img in glob.glob(IndyImageDirectory) if ".png" in img]
IndyMasks = [img for img in glob.glob(IndyMaskDirectory) if ".png" in img]

#################################################################################################

#loss_fn = FocalTverskyLoss()
loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
#loss_fn = smp.losses.JaccardLoss(smp.losses.BINARY_MODE, from_logits=True)
#loss_fn = smp.losses.TverskyLoss(smp.losses.BINARY_MODE, alpha=0.5, beta=0.5, from_logits=True)
#loss_fn = smp.losses.MCCLoss()
# Higher alpha puts more emphasis on punishing FP, beta for FN. Should sum up to 1

###################################################################################################
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

#This is a class for getting the dataset i.e. making the dataset available for use.
#This is not a dataloader in itself, it just makes the images available to use in the code

class CarvanaDataset(Dataset):
    def __init__(self, imgs, masks, transform=None):
        self.images = imgs
        self.masks = masks
        self.transform = transform




    def __len__(self):
        return(len(self.images))



    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)/255.0
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) # L = Luminance (Grayscale)
        # Change the mask to binary 1 and 0
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]


        return image, mask, img_path



##################################################################################################
from torch.utils.data import DataLoader

"""Creates the dataloader objects for the sets."""
def getLoaders(TrainImages, TrainMasks, TestImages, TestMasks, BatchSize, TrainTransform, TestTransform, Workers=4, pin_memory=True):

    train_ds = CarvanaDataset(imgs=TrainImages,masks=TrainMasks, transform=TrainTransform)
    train_loader = DataLoader(train_ds, batch_size=BatchSize, num_workers=Workers, pin_memory=pin_memory, shuffle=True)
    test_ds = CarvanaDataset(imgs=TestImages, masks=TestMasks, transform=TestTransform)
    test_loader = DataLoader(test_ds, batch_size=BatchSize, num_workers=Workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, test_loader

# def getIndyLoader(IndyImages, IndyMasks, BatchSize, Transform, Workers=4, pin_memory=True):
#     Indy_ds = CarvanaDataset(imgs=IndyImages, masks=IndyMasks, transform=Transform)
#     Indy_loader = DataLoader(Indy_ds, batch_size=BatchSize, num_workers=Workers, pin_memory=pin_memory, shuffle=False)
#
#     return Indy_loader
ImageCount = 0

"""Function to save a batch of test images to have a look at to see improvement."""
def savePredictionImagesTest(loader, model, out_folder, Epoch, BatchNumber, device="cuda"):
    #print("Saving Images")
    global ImageCount
    for x,y,z in loader:
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        grid = torchvision.utils.make_grid(tensor=preds, nrow=4)
        torchvision.utils.save_image(grid, f"{out_folder}/{ImageCount} Prediction.png")

        if not os.path.exists(f"{out_folder}/Epoch {0}, Batch {0} Ground Truth.png"):
            gtImages = torchvision.utils.make_grid(tensor=x, nrow=4)
            gtgrid = torchvision.utils.make_grid(tensor=y.unsqueeze(1), nrow=4)
            torchvision.utils.save_image(gtgrid, f"{out_folder}/Epoch {Epoch}, Batch {BatchNumber} Ground Truth.png")
            torchvision.utils.save_image(gtImages, f"{out_folder}/Epoch {Epoch}, Batch {BatchNumber} Originals.png")
        ImageCount+=1
        break

BestAcc = 0
BestLoss = 99999999
BestIOU = 0
BestDice = 0
BestMatthews = 0
"""Checks the accuracy periodically. Can be used for both training and testing"""
def checkAccuracy(loader, model, device="cuda", mode="Test"):
    global BestAcc
    global BestLoss
    global BestIOU
    global BestDice
    global BestMatthews
    correct = 0
    pixels = 0
    dice_score = 0
    lossbig = 0
    model.eval()
    batches = 0
    loop = loader
    Avg_IOU_Score = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0



    with torch.no_grad():
        for (data, targets, FilePath) in loop:
            if batches < TestBatches:

                x = data.to(Device)
                y = targets.to(Device).unsqueeze(1)
                predictions = model(x)
                loss = loss_fn(predictions, y)
                lossbig+=loss
                predictions = torch.sigmoid(predictions)
                predictions = (predictions > 0.5).float()
                correct += (predictions == y).sum()
                pixels += torch.numel(predictions)

                tp,fp,fn,tn = smp.metrics.get_stats(predictions.long(), y.long(), mode="binary")
                #print((smp.metrics.f1_score(tp,fp,fn,tn)))
                dice_score += (smp.metrics.f1_score(tp,fp,fn,tn, reduction="micro"))
                Avg_IOU_Score += (smp.metrics.iou_score(tp,fp,fn,tn, reduction="micro"))

                TP+=torch.sum(torch.sum(tp,dim=1)).item()
                FN+=torch.sum(torch.sum(fn,dim=1)).item()
                FP+=torch.sum(torch.sum(fp,dim=1)).item()
                TN+=torch.sum(torch.sum(tn,dim=1)).item()
                batches+=1
            else:
                break




    TestLosses_Batch.append(lossbig/TestBatches)
    try:
        Matthews = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    except:
        Matthews = 0
    if mode == "Test":
        if BestAcc < correct/pixels:
            torch.save(model.state_dict(), os.path.join(ResultsDirectory, "Best Model by accuracy.pt"))
            BestAcc = correct/pixels
            print(f"New accuracy record: {BestAcc}")
        if BestLoss > lossbig/TestBatches:
            torch.save(model.state_dict(), os.path.join(ResultsDirectory, "Best Model by loss.pt"))
            BestLoss = lossbig/TestBatches
            print(f"New loss record: {BestLoss}")
        if Avg_IOU_Score/TestBatches > BestIOU:
            torch.save(model.state_dict(), os.path.join(ResultsDirectory, "Best Model by IOU.pt"))
            BestIOU = Avg_IOU_Score / TestBatches
            print(f"New IOU record: {BestIOU}")
        if dice_score/TestBatches > BestDice:
            torch.save(model.state_dict(), os.path.join(ResultsDirectory, "Best Model by Dice.pt"))
            BestDice = dice_score / TestBatches
            print(f"New Dice record: {BestDice}")
        if Matthews > BestMatthews:
            torch.save(model.state_dict(), os.path.join(ResultsDirectory, "Best Model by Matthews.pt"))
            BestMatthews = Matthews
            print(f"New Matthews record: {BestMatthews}")




    model.train()
    return (correct/pixels), dice_score/len(loader), lossbig/TestBatches, Avg_IOU_Score/TestBatches, TP,TN,FP,FN

########################################################################################################################


#######################################################################################################################
# Create a function for training SegNet for one epoch
def TrainSegNET(dataloader, testload, model, optimiser, loss_fn, scaler, ep, freq):



    # Get the data from the dataloader and move to the GPU if available
    loop = tqdm(dataloader, position=0, leave=True)
    for batch_idx, (data, targets, FilePath) in enumerate(loop):


        data = data.to(Device)
        targets = targets.float().unsqueeze(1).to(Device)

        # Run the images through the model, get the output binary masks and compare to the ground truth masks
        with torch.cuda.amp.autocast():
            predictions = model(data)

            loss = loss_fn(predictions, targets)
            #print(loss)

        # Backpropigate
        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        out_folder = os.path.join(ResultsDirectory, "OutputImages")

        try:
            os.mkdir(out_folder)
        except:
            pass

        freq = int(len(loop)/10)





        if batch_idx % freq == 0:
            savePredictionImagesTest(testload, model, out_folder, ep, BatchNumber=batch_idx, device=Device)

        # After n training batches, run the test set through


            #print("Getting Test Accuracy")
            Acc, DC, loss, IOU, TP,TN,FP,FN = checkAccuracy(testload, model, device=Device)
            #print("Getting Training Accuracy")
            AccTrain, DCTrain, trainloss, trainIOU, TTP,TTN,TFP,TFN = checkAccuracy(dataloader, model, device=Device, mode="Train")
            TestAccuracies_Batch.append(Acc.item())
            TrainAccuracies_Batch.append(AccTrain.item())
            Dice_Score_Batch.append(DC.item())
            TrainLosses_Batch.append(trainloss.item())
            TestLosses_Batch.append(loss.item())
            BatchIDs.append(ep + (batch_idx * (1/len(loop))))

            with open(ResultsDirectory+"\\Segmentation Results.csv", "a") as f:
                f.write(f"{ep},{batch_idx},{AccTrain},{DCTrain},{trainloss},{trainIOU},{TFN},{TFP},{TTN},{TTP},{Acc},{DC},{loss},{IOU},{FN},{FP},{TN},{TP}\n")


            loop.set_postfix(Training_Loss=trainloss.item(), Test_Loss=loss.item(), Train_Acc=AccTrain.item(), Test_Acc=Acc.item() )

        # Update TQDM loop
    TestLosses_Epoch.append(loss.item())
    EpochIDs.append(ep + 1)

def RunSegNET():

    # Define the transformations for the training and test images
    #print("Loading Initial Datasets")
    train_transform = A.Compose([#A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
                                 ToTensorV2()])

    test_transform = A.Compose([#A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
                                ToTensorV2()])



    train_loader, test_loader = getLoaders(TrainImages, TrainMasks, TestImages, TestMasks, BatchSize, train_transform, test_transform)
    #Indy_loader = getIndyLoader(IndyImages,IndyMasks, BatchSize, test_transform)
    #print("Initial Datasets loaded")
    for epoch in range(Epochs):
        # if epoch == 0:
        #     SampleFreq = 500
        # elif 1 <= epoch <= 3:
        #     SampleFreq = 200
        # else:
        #     SampleFreq = 50

        SampleFreq=60


        loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

       # print("Epoch ", epoch)
        TrainSegNET(dataloader=train_loader, testload=test_loader, model=model, optimiser=optimiser, loss_fn=loss_fn, scaler=Scaler, ep=epoch, freq=SampleFreq)

        # Save the model after every epoch
        torch.save(model.state_dict(), f"efficientnet-b0 Epoch {epoch}.pt")

        # Check the accuracy
        #Acc, DC, loss = checkAccuracy(test_loader, model, device=Device)
        #TestAccuracies_Epoch.append(Acc.item())
        #Dice_Score_Epoch.append(DC.item())



if  __name__ == "__main__":
    with open(ResultsDirectory + "\\TLS Segmentation Results.csv", "a") as f:
        f.write(f"Epoch	Batch,Epoch Fraction,Training Accuracy,Training Dice Score,Training Loss,Training IOU,Train FN,Train FP,Train TN,Train TP,Test Acc,Test Dice Score,Test Loss,Test IOU,Test FN,Test FP,Test TN,Test TP\n")


    RunSegNET()


############################################ Make Acc oss CSV ########################################




    import pandas as pd

    data = {"Batch ID": BatchIDs, "Train Batch Acc":TrainAccuracies_Batch, "Train Batch loss":TrainLosses_Batch, "Test Batch Acc": TestAccuracies_Batch, "Test Batch Loss":TestLosses_Batch, "Test Batch Dice Score": Dice_Score_Batch,
            "Epoch": EpochIDs, "Epoch Acc":TestAccuracies_Epoch, "Epoch Loss": TestLosses_Epoch, "Epoch Dice Score": Dice_Score_Epoch}

    df = pd.DataFrame.from_dict(data, orient='index').T

    df.to_csv(path_or_buf="efficientnet-b0 Training.csv", index=False, index_label=None)
    #os.system("shutdown /s")