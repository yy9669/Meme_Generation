import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

def get_loader(image_path, metadata_path, crop_size, image_size, batch_size, mode='train'):
    """Build and return data loader."""

    if mode == 'train':
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = ImageFolder(image_path, transform)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader