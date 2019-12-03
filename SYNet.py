#!/usr/bin/env python
# coding: utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models



class SYNet(nn.Module):
    """Container module for SYNet"""
    def __init__(self, params):
        super(SYNet, self).__init__()
        pass

    def init_weights(self):
        pass

    def forward(modelsself, input, hidden):
        pass
    def init_hidden(self, bsz):
        pass


def Baseline(model_name, train_all):
    i = 0
    if(model_name == 'resnet18'):
        model = models.resnet18(pretrained = True)
        for child in model.children():
            i += 1
            if i <= train_all:
                for param in child.parameters():
                    param.required_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)

    elif(model_name == 'resnet50'):
        model = models.resnet50(pretrained = True)
        for child in model.children():
            i += 1
            if i <= train_all:
                for param in child.parameters():
                    param.required_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)

    elif(model_name == 'resnet101'):
        model = models.resnet101(pretrained = True)
        for child in model.children():
            i += 1
            if i <= train_all:
                for param in child.parameters():
                    param.required_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)

    elif(model_name == 'vgg16'):
        if(train_all):
            model = models.vgg16(pretrained=True)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, 5, bias=True)
        else:
            model = models.vgg16(pretrained = True)
            for param in model.parameters():
                param.required_grad = False

            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, 5, bias=True)

    elif(model_name == 'densenet121'):
        if(train_all):
            model = models.densenet121(pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 5, bias=True)
        else:
            model = models.densenet121(pretrained = True)
            for param in model.parameters():
                param.required_grad = False

            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 5, bias=True)

    elif(model_name == 'mnasnet1_0'):
         if(train_all):
            model = models.mnasnet1_0(pretrained=True)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 5, bias=True)
         else:
            model = models.mnasnet1_0(pretrained = True)
            for param in model.parameters():
                param.required_grad = False

            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 5, bias=True)
    return model
