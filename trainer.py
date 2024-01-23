from dataset import AnimalsDataset
from torch.utils.data import DataLoader
from transformations import get_tfms

class Trainer:

    def _train_step(self):
        pass

    def _valid_step(self):
        pass

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
                                      )
