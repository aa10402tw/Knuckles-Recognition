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

def get_InceptionV3(pretrained='imagenet'):
    ## Load the model 
    model_conv = torchvision.models.inception_v3(pretrained=pretrained)
    model_conv.aux_logits = False

    ## Lets freeze the first few layers. This is done in two stages 
    # Stage-1 Freezing all the layers 
    if freeze_layers:
        for i, param in model_conv.named_parameters():
            param.requires_grad = True

    # Since imagenet as 1000 classes , We need to change our last layer according to the number of classes we have,
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, n_class)

    # Stage-2 , Freeze all the layers till "Conv2d_4a_3*3"
    ct = []
    for name, child in model_conv.named_children():
        if "Conv2d_4a_3x3" in ct:
            for params in child.parameters():
                params.requires_grad = True
        ct.append(name)
    
    model_conv.cuda()
    print("[Using inception_v3]")
    
    return model_conv