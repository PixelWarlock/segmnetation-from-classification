import torch
from dataset import AnimalsDataset
from torch.utils.data import DataLoader
from classification_model import Classifier
from full_model import FullModel
from transformations import get_tfms
from segmentation_model import get_unet
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:

    @staticmethod
    def step(model, dataloader, criterion, optimizer, mode):
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

        elif mode == 'valid':
            model.eval()
            with torch.no_grad():
                for img, label in dataloader:
                    img = img.to(device).type(torch.cuda.FloatTensor)
                    label = label.to(device).type(torch.cuda.FloatTensor)

                    probs = model(img)
                    loss  = criterion(probs, label)
                    losses.append(loss.item())
            model.train()
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
        classifier = Classifier(num_classes=5)

        full_model = FullModel(
            segmentor=segmentor,
            classifier=classifier
        ).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(full_model.parameters(), lr=0.0003)

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
            
            print(f"Epoch: {epoch} | Train loss: {np.mean(train_losses)} | Valid loss: {np.mean(valid_losses)}")
            print(100*'-')
            


