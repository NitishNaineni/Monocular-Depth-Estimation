from Data import Depthset, Agument, toTensor
from models import Model

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

import mat73
import kornia
import pickle
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

file_path1 = "nyu_depth_data_labeled.mat"
data_dict1 = mat73.loadmat(file_path1)

file_path2 = "nyu_depth_v2_labeled.mat"
data_dict2 = mat73.loadmat(file_path2)

images1 = data_dict1['images']
images2 = data_dict2['images']
depths1 = data_dict1['depths']
depths2 = data_dict2['depths']
images = np.concatenate([images1,images2],axis=3)
depths = np.concatenate([depths1,depths2],axis=2)

x = 2986
train_images = images[:,:,:,:x]
valid_images = images[:,:,:,x:-200]
test_images = images[:,:,:,-200:]
train_depths = depths[:,:,:x]
valid_depths = depths[:,:,x:-200]
test_depths = depths[:,:,-200:]

data_dict_train = {"images":train_images,"depths":train_depths}
data_dict_valid = {"images":valid_images,"depths":valid_depths}
data_dict_test = {"images":test_images,"depths":test_depths}


print("data loaded")

train_dataset = Depthset(
    data_dict=data_dict_train,
    transform=transforms.Compose([Agument(probability=0.5),toTensor()])
    )

valid_dataset = Depthset(
    data_dict=data_dict_valid,
    transform=transforms.Compose([Agument(probability=0.5),toTensor()])
    )

test_dataset = Depthset(
    data_dict=data_dict_test,
    transform=transforms.Compose([Agument(probability=0.5),toTensor()])
    )

print("dataset created")

model = Model().cuda()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

epochs = 1000
lr = 0.0001
batchsize = 8

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batchsize,
    shuffle=True
    )

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=batchsize,
    shuffle=False
    )

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batchsize,
    shuffle=False
    )

optimizer = torch.optim.Adam(model.parameters(),lr)
criterion = nn.L1Loss()
ssim = kornia.losses.SSIMLoss(window_size = 11, max_val = 1000.0 / 10.0, reduction='none')

losses = []
loss_epoch = []
valid_losses = []
best_val_loss = float('inf')

for epoch in range(epochs):
    cum_loss = 0
    
    model.train()
    start = time.time()
    for i,batch in enumerate(train_loader):
        optimizer.zero_grad()

        image = batch['image'].cuda()
        depth = batch['depth'].cuda()

        depth = 1000.0 / depth

        out = model(image)

        loss = criterion(out, depth)
        l_ssim = torch.clamp((1 - ssim(out, depth)) * 0.5, 0, 1)
        loss = (1.0 * l_ssim.mean().item()) + (0.1 * loss)

        losses.append(loss.item())
        cum_loss += loss.item()

        loss.backward()
        optimizer.step()

    loss_epoch.append(cum_loss)

    with open("./trained_models/epoch_loss_log",'wb') as fp:
        pickle.dump(loss_epoch,fp)

    
    valid_loss = 0.0
    model.eval()
    for i, batch in enumerate(valid_loader):
        image = batch['image'].cuda()
        depth = batch['depth'].cuda()
        
        target = model(image)
        
        loss = criterion(target, depth)
        l_ssim = torch.clamp((1 - ssim(target, depth)) * 0.5, 0, 1)
        loss = (1.0 * l_ssim.mean().item()) + (0.1 * loss)
        
        valid_loss += loss.item()
        
    valid_losses.append(valid_loss)
    
    with open("./trained_models/valid_loss_log",'wb') as fp:
        pickle.dump(valid_losses,fp)
        
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        path = f"./trained_models/{epoch}.pth"
        torch.save(model.state_dict(),path)
        print("model saved")
        

    end = time.time()

    print(f"[{epoch}/{epochs}] train_loss: {cum_loss:.3f} valid_loss: {valid_loss:.3f} time: {(end - start):.3f}")


model.eval()
test_loss = 0
for i, batch in enumerate(test_loader):
    image = batch['image'].cuda()
    depth = batch['depth'].cuda()
    
    target = model(image)
    
    loss = criterion(target, depth)
    l_ssim = torch.clamp((1 - ssim(target, depth)) * 0.5, 0, 1)
    loss = (1.0 * l_ssim.mean().item()) + (0.1 * loss)
    
    test_loss += loss.item()

print(f"test loss: {test_loss}")
    
