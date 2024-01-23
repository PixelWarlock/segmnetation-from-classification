from argparse import ArgumentParser
from trainer import Trainer

def train(args):
    Trainer.run(**args.__dict__)

def inference():
    pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-sm', '--segmentation_model', type=str, default='unet')
    parser.add_argument('-cm', '--classification_model', type=str, default='efficientnet_b0')
    parser.add_argument('-td', '--train_dataset', type=str, default='/media/kduraj/Nowy/X/datasets/classification/animals/train')
    parser.add_argument('-vd', '--validation_dataset', type=str, default='/media/kduraj/Nowy/X/datasets/classification/animals/val')
    
    args = parser.parse_args()
    train(args)