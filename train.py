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

import sklearn.metrics
import itertools

from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/nonupsampled_resnet18')

log = ""
class_names = ["concrete_cement","healthy_metal","incomplete","irregular","other"]

def plot_confusion_matrix(cm, class_names):
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.YlGn)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Normalize the confusion matrix.
  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "gray" if cm[i, j] > threshold else "black"
    plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

def log_confusion_matrix(epoch, output, label):
  # Calculate the confusion matrix.
  cm = sklearn.metrics.confusion_matrix(label, output)
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm, class_names=class_names)

  # Log the confusion matrix as an image summary.
  writer.add_figure('Valid/confusion_matrix', figure, epoch)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, backbone=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_step = -1
        self.backbone = backbone

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print("current min val loss:", self.val_loss_min)
            if self.counter >= int(self.patience):
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        if self.verbose:
            print(f'Validation loss decreased at epoch {epoch} ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), '{}_checkpoint.pt'.format(self.backbone))
        self.val_loss_min = val_loss
        self.best_step = epoch 

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


def load_data(batch_size, upsampling, retrain, options):

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
    
    scale = 299 if options.pretrained_name=='inception' else 224 

    transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(degrees = 90, resample = False, expand = True),
        transforms.RandomVerticalFlip(),
        transforms.Resize((scale,scale)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print('Length of given dataset : {}'.format(len(train_images))) 
    validation_images = train_images[::5]
    validation_images_labels = train_images_labels[::5]

    if not retrain:
        indices = [i for i in range(len(train_images)) if (i) % 5]
        train_images = [train_images[i] for i in indices]
        train_images_labels = [train_images_labels[i] for i in indices]
    
    train_dataset = RoofDataset(train_images, 
                                train_images_labels,
                                transformations)
    train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0
                         )

    valid_transforms = transforms.Compose([
        transforms.Resize((scale,scale)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
    valid_dataset = RoofDataset(validation_images,
                                validation_images_labels,
                                valid_transforms)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0)
    print('Length of train_loader : {} | Length of valid_loader : {}'.format(len(train_loader.dataset), len(valid_loader.dataset)))
    return train_loader, valid_loader

def train(epochs, train_loader, valid_loader, options, best_step=None):

    global log
    model = SYNet.Baseline(options.pretrained_name, eval(options.train_all))
    model.cuda()
    optimizer = optim.SGD(model.parameters(), 
                          lr=float(options.lr), 
                          momentum=float(options.momentum),
                          weight_decay=float(options.weight_decay))
    
    #class_weights = [7381/1387, 7381/7381, 7381/668, 7381/5241, 7381/193]
    #if eval(options.upsampling):
    #    class_weights = [7381/1387, 7381/7381, 7381/(668 * 2), 7381/5241, 7381/(193 * 5)]
    #class_weights = torch.FloatTensor(class_weights)
    #criterion = nn.CrossEntropyLoss(weight=class_weights.cuda())
    criterion = nn.CrossEntropyLoss()

    train_tag = 'Train'
    valid_tag = 'Valid'

    if best_step is None:
        early_stopping = EarlyStopping(patience=options.patience, verbose=True, backbone=options.pretrained_name)

    if best_step is not None:
        epochs = best_step
        train_tag = "Retrain"
        valid_tag = "Revalid"

    for epoch in range(1, epochs + 1):
        train_losses = []
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            data = data.cuda(async=True)
            target = target.cuda(async=True)
            optimizer.zero_grad()
            output = model(data)
            # Output format different for inception
            output = output[0] if options.pretrained_name == 'inception' else output
            _, idx = torch.max(target,1)
            loss = criterion(output, idx)
            train_losses.append(loss.item())
            loss.backward()
            optimizer`.step()
            if batch_idx % 30 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data))
                writer.add_scalar('{}/batch_loss'.format(train_tag), 
                                  loss.item(), 
                                  (epoch - 1) * len(train_loader) + batch_idx)
        valid_losses = []
        nll_losses = []
        predlist = torch.zeros(0,dtype=torch.long, device='cpu')
        labellist = torch.zeros(0,dtype=torch.long, device='cpu')
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                model.eval()
                data, target = data.cuda(), target.cuda()
                # Output format different for inception
                output = model(data)
                output = output[0] if options.pretrained_name == 'inception' else output
                _, idx = torch.max(target, 1)
                _, pred = torch.max(output, 1)

                predlist=torch.cat([predlist,pred.view(-1).cpu()])
                labellist = torch.cat([labellist,idx.view(-1).cpu()])

                loss = criterion(output, idx)
                subloss = nn.functional.nll_loss(nn.functional.log_softmax(output, dim=1), idx)
                valid_losses.append(loss.item())
                nll_losses.append(subloss.item())
                if batch_idx % 30 == 0:
                    writer.add_scalar('{}/batch_loss'.format(valid_tag),
                                      loss.item(),
                                      (epoch - 1) * len(valid_loader) + batch_idx)
        if best_step is None and epoch % 10 == 0:
            log_confusion_matrix(epoch, predlist.numpy(), labellist.numpy())
        print('\nEpoch: [{} / {}], Train Loss: {} | Validation Loss: {}\n'.format(epoch, 
                                                             epochs,
                                                             np.average(train_losses),
                                                             np.average(valid_losses)))
        print("nllLoss: {}\n".format(np.average(nll_losses)))
        writer.add_scalar('{}/epoch_loss'.format(train_tag),
                          np.average(train_losses),
                          epoch)
        writer.add_scalar('{}/epoch_loss'.format(valid_tag),
                          np.average(valid_losses),
                          epoch)
        writer.add_scalar('{}/epoch_nll_loss'.format(valid_tag),
                          np.average(nll_losses),
                          epoch)

        if best_step is None:
            early_stopping(np.average(valid_losses), model, epoch)
        
            if early_stopping.early_stop:
                print("Early Stopping")
                break
    
    if best_step is None:
        best_step = early_stopping.best_step
        log += "Best step achieved: {}\n".format(best_step)
        log += "Best valloss achieved: {}\n".format(early_stopping.val_loss_min)
        model.load_state_dict(torch.load('{}_checkpoint.pt'.format(options.pretrained_name)))
        print("Best model loaded from checkpoint successfully")
    log += "Final Validation Error achieved: {}\n".format(np.average(valid_losses))
    return model, best_step


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
    parser.add_argument('--retrain', dest='retrain',
    default=False, help='whether to retrain on best step acquired from early Stopping')
    parser.add_argument('--patience', dest='patience',
    default=30, help='patience value for early stopping')
    parser.add_argument('--note', dest='note',
    required=True, help="brief description of model being trained")    

    options = parser.parse_args()
    
    print(options.pretrained_name)
 
    global log
 
    log += "=====================================\n"
    log += "{}\n".format(options.note)
    log += "Backbone used: {}\n".format(options.pretrained_name)
    log += "Model name to be saved: {}\n".format(options.model_name)
    log += "upsampling used?: {}\n".format(options.upsampling)
    log += "Retrain after early stopping? {}\n".format(options.retrain)
    log += "How many Layers are going to be frozen? {}\n".format(options.train_all)
    log += "\nHyperparameters Used:: \n"
    log += "Batch size: {} | Learning Rate: {}\n".format(options.batch_size, options.lr)
    log += "Momentum : {} | Weight decay : {}\n".format(options.momentum, options.weight_decay)
    log += "Epoch: {} | Patience : {}\n".format(options.epoch, options.patience)
    train_loader, valid_loader  = \
        load_data(int(options.batch_size), eval(options.upsampling), False, options)
    log += "\n Training summary: \n"
    model, best_step = train(int(options.epoch), train_loader, valid_loader, options)
    
    torch.save(model.state_dict(),
        './models/{}'.format(options.model_name))

    print("Model {} has been saved successfully after {} epochs.".format(
        options.model_name,
        best_step))

    if eval(options.retrain):
        log += "\n Retraining summary: \n"
        print("\nRetraining INCLUDING valset, up to {} epochs".format(best_step))
        train_loader, valid_loader = \
            load_data(int(options.batch_size), eval(options.upsampling), True, options)
        model, _ = train(int(options.epoch), train_loader, valid_loader, options, best_step)
 
        torch.save(model.state_dict(), 
                   './models/retrained_{}'.format(options.model_name))
        print("Retrained model has been saved. {}".format('./models/retrained_{}'.format(options.model_name)))
    
    log += "=====================================\n"

    with open ("log.txt", "a") as f:
        f.write(log)

if __name__ == '__main__':
    main()
