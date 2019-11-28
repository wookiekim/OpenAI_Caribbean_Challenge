#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import glob
import time

from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms, utils, datasets

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import SYNet

class RoofDataset(Dataset):
    def __init__(self, train_image_paths, train_images_labels,transform=None):
        self.image_paths = train_image_paths
        self.image_labels = train_images_labels
        self.transform=transform
        
    def __getitem__(self, index):
        roof_image = Image.open(self.image_paths[index])
        roof_image = roof_image.convert('RGB')
        if self.transform is not None:
            roof_image = self.transform(roof_image)
        
        material_type = torch.LongTensor(self.image_labels[index])
        return roof_image, material_type    
    
    def __len__(self):
        return len(self.image_paths)


def load_data(batch_size, upsampling):

    # Define Class Labels
    concrete_cement_type = [1.0, 0.0, 0.0, 0.0, 0.0]
    healthy_metal_type = [0.0, 1.0, 0.0, 0.0, 0.0]
    incomplete_type = [0.0, 0.0, 1.0, 0.0, 0.0]
    irregular_metal_type = [0.0, 0.0, 0.0, 1.0, 0.0]
    other_type = [0.0, 0.0, 0.0, 0.0, 1.0]

    #1387
    concrete_cement_images = glob.glob('./training/unmasked/concrete_cement/*.png')
    a= [concrete_cement_type] * len(concrete_cement_images)
    #7381
    healthy_metal_images = glob.glob('./training/masked/healthy_metal/*.png')
    b=[healthy_metal_type] * len(healthy_metal_images)
    #668
    incomplete_images = glob.glob('./training/masked/incomplete/*.png')
    if upsampling:
        incomplete_images = incomplete_images * 2
    c=[incomplete_type] * len(incomplete_images)
    #5241
    irregular_metal_images = glob.glob('./training/masked/irregular_metal/*.png')
    d=[irregular_metal_type]  * len(irregular_metal_images)
    #193
    other_images = glob.glob('./training/masked/other/*.png')
    if upsampling:
        other_images = other_images * 5
    e=[other_type] * len(other_images)

    train_images = [concrete_cement_images, healthy_metal_images, incomplete_images, irregular_metal_images, other_images]
    train_images_labels = [a,b,c,d,e]

    train_images = [item for sublist in train_images for item in sublist]
    train_images_labels = [item for sublist in train_images_labels for item in sublist]

    transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(degrees = 90, resample = False, expand = True),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness = 0.2, contrast = 0.1),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    train_dataset = RoofDataset(train_images, train_images_labels,transformations)
    train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0
                         )
    return train_loader

def train(epochs, train_loader, options):
    model = SYNet.Baseline(options.pretrained_name, eval(options.train_all))
    model.cuda()
    optimizer = optim.SGD(model.parameters(), 
                          lr=float(options.lr), 
                          momentum=float(options.momentum),
                          weight_decay=float(options.weight_decay))
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.cuda(async=True)
            target = target.cuda(async=True)
            optimizer.zero_grad()
            output = model(data)
            _, idx = torch.max(target,1)
            loss = criterion(output, idx)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data))

    return model


def main():
    parser =  argparse.ArgumentParser()

    parser.add_argument('--batch_size', dest='batch_size',
    default=32, help='batch size for training')
    parser.add_argument('--learning_rate', dest='lr',
    default=0.0001, help='(initial)learning rate for training')
    parser.add_argument('--momentum', dest='momentum',
    default=0.5, help='momentum for optimizer')
    parser.add_argument('--weight_decay', dest='weight_decay',
    default=0.01, help='value of weight decay')
    parser.add_argument('--epoch', dest='epoch',
    default=50, help='epochs for training')
    parser.add_argument('--model_name', dest="model_name",
    required=True, help="name of model to be saved to .pt file")
    parser.add_argument('--pretrained_name', dest='pretrained_name',
    required=True, help="name of pretrained model to be used for transfer learning")
    parser.add_argument('--train_all', dest='train_all',
    default=False, help='whether to finetune on all layers')    
    parser.add_argument('--upsampling', dest='upsampling',
    default=True, help='whether to use upsampling or not')
 
    options = parser.parse_args()
 
    train_loader = load_data(int(options.batch_size), eval(options.upsampling))
    
    model = train(int(options.epoch), train_loader, options)

    torch.save(model.state_dict(), 
               './models/{}'.format(options.model_name)
              )
    print("Model {} has been saved successfully after {} epochs.".format(
           options.model_name,
           options.epoch))

if __name__ == '__main__':
    main()
