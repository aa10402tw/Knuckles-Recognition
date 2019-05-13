import torch

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

import time
import os
import glob

from tqdm import tqdm as tqdm

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import DataLoader

import torch
import numpy as np
import torchvision
import os

from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

### Global Parameter 
train_transform = transforms.Compose([
    transforms.Resize(299), # 299 for inception / 224 for resnet
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


USE_CUDA = True
freeze_layers = True
n_class = 14

import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import argparse
from PIL import Image
import torch.nn.functional as F

from model import *

def train_model(model, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        
        pbar = tqdm(total=len(train_loader), unit=' batches')
        #pbar.set_description('Epoch %i/%i (Training)' % (epoch+1, epochs))
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        for step, (inputs, y_true) in enumerate(train_loader):
            
            if USE_CUDA:
                x_sample, y_true = inputs.cuda(), y_true.cuda()
            x_sample, y_true = Variable(x_sample), Variable(y_true)
            
            # parameter gradients들은 0값으로 초기화 합니다.
            optimizer.zero_grad()
            # Feedforward
            y_pred = model(x_sample)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            
            # acc
            pred_y = y_pred.data.cpu()
            target_y = y_true.data.cpu().numpy()
            _, pred_y = torch.max(pred_y, 1)
            pred_y = pred_y.data.numpy()
            epoch_total += 32 
            epoch_correct += sum(pred_y == target_y)
            pbar.set_postfix( {'acc': '%.2f'%(epoch_correct/epoch_total) } )
            pbar.update()
        time.sleep(0.1)
        # print('[%i] loss: %.4f}' %(epoch+1, epoch_loss/step))
        pbar.close()

def validate(model, epochs=1):
    model.train(False)
    n_total_correct = 0
    pbar = tqdm(total=len(test_loader), unit=' batches')
    for step, (inputs, y_true) in enumerate(test_loader):
        if USE_CUDA:
            x_sample, y_true = inputs.cuda(), y_true.cuda()
            
        x_sample, y_true = Variable(x_sample), Variable(y_true)
        y_pred = model(x_sample)
        _, y_pred = torch.max(y_pred.data, 1)

        n_correct = torch.sum(y_pred == y_true.data)
        n_total_correct += n_correct
        pbar.update()
    n_total_correct = n_total_correct.data.cpu().numpy()
    print('val correct ratio : %i/%i'%(n_total_correct, len(test_loader.dataset)))
    print('val accuracy:', n_total_correct/len(test_loader.dataset))
    print()
    pbar.close()
    return  n_total_correct/len(test_loader.dataset)

df_train = pd.read_csv('train_data_info.csv', sep='\t')
df_val = pd.read_csv('val_data_info.csv', sep='\t')
df_allData = pd.read_csv('allData_info.csv', sep='\t')

num_imgs = len(df_allData)
print((set(df_allData['particpant'].values)))

names_list = list(set(df_allData['particpant'].values))
np.random.shuffle(names_list)
train_names = names_list[:8]
val_names = names_list[8:]
print('train:', train_names)
print('val:', val_names)
folder_name = 'basic_inceptionV3_data'

train_data = datasets.ImageFolder('%s/train/'%folder_name, train_transform)
test_data = datasets.ImageFolder('%s/val/'%folder_name, test_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

model = get_InceptionV3()

optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.001, momentum=0.99)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# train_model(model, criterion, optimizer, epochs=1)
acc = validate(model)
print('Accuracy :', acc)print(np.array(interval_list).mean())