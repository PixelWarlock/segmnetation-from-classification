import os
import torch
import pandas as pd
from dataset import AnimalsDataset
from torch.utils.data import DataLoader
from classification_model import get_efficientnet_b0 #Classifier
from full_model import FullModel
from transformations import get_tfms
from segmentation_model import get_unet
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:

    @staticmethod
    def step(model, dataloader, criterion, optimizer, mode):
        pbar = tqdm(total=dataloader.__len__())

        losses = list()
        if mode == 'train':
            for img, label in dataloader:
                optimizer.zero_grad()

                img = img.to(device).type(torch.cuda.FloatTensor)
                label = label.to(device).type(torch.cuda.FloatTensor)

                probs = model(img)
                loss  = criterion(probs, label)
                
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                pbar.update()

        elif mode == 'valid':
            model.eval()
            with torch.no_grad():
                for img, label in dataloader:
                    img = img.to(device).type(torch.cuda.FloatTensor)
                    label = label.to(device).type(torch.cuda.FloatTensor)

                    probs = model(img)
                    loss  = criterion(probs, label)
                    losses.append(loss.item())
                    pbar.update()
            model.train()
        pbar.close()
        return losses

    def run(batch_size:int,
            epochs:int,
            segmentation_model:str,
            classification_model:str,
            train_dataset:str,
            validation_dataset:str):
        
        tfms = get_tfms()

        train_dataset = AnimalsDataset(src=train_dataset, tfms=tfms)
        validation_dataset = AnimalsDataset(src=validation_dataset, tfms=tfms)

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
        
        valid_dataloader = DataLoader(dataset=validation_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)

        segmentor = get_unet(in_channels=3, out_channels=1)
        classifier = get_efficientnet_b0(num_classes=5) #Classifier(num_classes=5, in_channels=3)

        full_model = FullModel(
            segmentor=segmentor,
            classifier=classifier
        ).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(full_model.parameters(), lr=0.0005)
        df = pd.DataFrame(columns=['Epoch', 'Train loss', 'Valid loss'])

        for epoch in range(epochs):
            train_losses=Trainer.step(model=full_model,
                         dataloader=train_dataloader,
                         criterion=criterion,
                         optimizer=optimizer,
                         mode='train')
            
            valid_losses=Trainer.step(model=full_model,
                         dataloader=valid_dataloader,
                         criterion=criterion,
                         optimizer=optimizer,
                         mode='valid')
            
            train_loss = np.mean(train_losses)
            valid_loss = np.mean(valid_losses)

            print(f"Epoch: {epoch} | Train loss: {train_loss} | Valid loss: {valid_loss}")
            print(100*'-')

            df.loc[len(df.index)] = [epoch, train_loss, valid_loss]

            torch.save(full_model.classifier.state_dict(), os.path.join(os.getcwd(), f"models/classifiers/classifier_{epoch}.pt"))
            torch.save(full_model.segmentor.state_dict(), os.path.join(os.getcwd(), f"models/segmentors/segmentor_{epoch}.pt"))

        df.to_csv(os.path.join(os.getcwd(), "results.csv"))
