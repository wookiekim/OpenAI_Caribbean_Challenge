#!/usr/bin/env python
# coding: utf-8

import argparse

from torch.utils.data import Dataset, DataLoader
import pandas as pd

from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import  transforms
import SYNet
import time


class RoofTestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform=transform
 
    def __getitem__(self, index):
        roof_id = self.image_paths[index].replace('./testing/masked/', '').replace('.png', '')
        roof_image = Image.open(self.image_paths[index])
        roof_image = roof_image.convert('RGB')

        if self.transform is not None:
            roof_image = self.transform(roof_image)
        return roof_image, roof_id

    def __len__(self):
        return len(self.image_paths)

def load_data():

    test_images = list(pd.read_csv("submission_format.csv",index_col=0).index.values)
    test_images = ['./testing/masked/' + str(i) + '.png' for i in test_images]

    transformations = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    test_dataset = RoofTestDataset(test_images, transformations)
    test_loader = DataLoader(test_dataset,
                        batch_size = 1,
                        num_workers = 0)

    return test_loader


def test(model, test_loader):
    for p in model.parameters():
        if p.grad is not None:
            del p.grad
    torch.cuda.empty_cache()
    
    test_id = []
    test_label = []
    list_from_labelTensor = []

    model.eval()
    predictions = []
    for data, roof_id in test_loader:
        replaced_roof_id = roof_id[0]
        test_id.append(replaced_roof_id)
        data = data.cuda(async=True) # On GPU
        output_res = model(data)
        list_from_labelTensor = \
            torch.nn.functional.softmax(output_res, dim=1).tolist()
        test_label.append(list_from_labelTensor)

    return test_id, test_label
        
def pred_to_csv(test_id, test_label, model_name):
    submission_dict = {}
    submission_dict['id'] = []
    submission_dict['concrete_cement'] = [] 
    submission_dict['healthy_metal'] = []
    submission_dict['incomplete'] = []
    submission_dict['irregular_metal'] = []
    submission_dict['other'] = []

    for i, j in zip(test_id, test_label):
        submission_dict['id'].append(i)
        submission_dict['concrete_cement'].append(j[0][0])
        submission_dict['healthy_metal'].append(j[0][1])
        submission_dict['incomplete'].append(j[0][2])
        submission_dict['irregular_metal'].append(j[0][3])
        submission_dict['other'].append(j[0][4])
    
    csv_filename = model_name.split('.')[0] + '.csv'    

    pd.DataFrame(submission_dict).to_csv( 
        "./submissions/{}".format(csv_filename),
        index = False)
    print("Prediction results have been successfully saved to {}".format(
        "./submissions/{}".format(csv_filename)))

def main():
    parser =  argparse.ArgumentParser()

    parser.add_argument('--model_name', dest="model_name",
    required=True, help="name of model to be retrieved from .pt file")
    parser.add_argument('--pretrained_name', dest="pretrained_name",
    required=True, help="name of pretrained model being used")

    options = parser.parse_args()
 
    model = SYNet.Baseline(options.pretrained_name, True)
    model.load_state_dict(torch.load('./models/{}'.format(options.model_name))) 
    model.cuda()

    test_loader = load_data()    

    test_id, test_label = test(model, test_loader)

    pred_to_csv(test_id, test_label, options.model_name)

if __name__ == '__main__':
    main() 
