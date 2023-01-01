# this file is based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""Train a backdoored model on the YouTube-Face dataset."""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy 
import torch.nn.functional as F
import numpy as np
from new_vgg_face2 import *
from tqdm import tqdm
import argparse
from save_image import save_image 
from data_loader import get_test_loader, get_backdoor_loader
import random


# process the data
def data_process(batch_size=256):
    mean = [0.367035294117647,0.41083294117647057,0.5066129411764705] # the mean value of vggface dataset 
    data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),

    'test': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
    
    }
                                
    data_dir = '/scratch/gpfs/tw6664/data/youtubeface/Images/'   # change this if the data is in different loaction 
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train','test']}


    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True)
              for x in ['train', 'test']}


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    return dataloaders,dataset_sizes



def get_arguments():
    parser = argparse.ArgumentParser()
    # backdoor attacks
    parser.add_argument('--epochs', type=int, default=50, help='epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--inject_portion', type=float, default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_type', type=int, default=90, help='degree of backdoor trigger')
    parser.add_argument('--name', type=str, default='1', help='name of the experiment')
    parser.add_argument('--seed', type=int, default=0, help='seed for random')
    parser.add_argument('--trans','--ts', type=int, default=0, help='trans = 1,2,3,4,5,6: random_rotate: 0,15,30,45,90,180')
    return parser


def train_model(model, criterion, optimizer, scheduler, num_epochs, opt):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    print("start")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test_clean','test_bad' ]:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.

            for idx , (inputs, labels) in  enumerate(tqdm(dataloaders[phase])):
                device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                inputs = inputs.to(device1)
                labels = labels.to(device1)
                #print(inputs.size(),labels.size())
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) 
                    
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'test_clean' and epoch_acc >= best_acc:


                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if phase == 'train':
            scheduler.step()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model





if __name__ == "__main__":
    opt = get_arguments().parse_args()

    torch.manual_seed(opt.seed)
    np.random.seed(0)
    random.seed(0)


    train_loader,train_loader_size = get_backdoor_loader(opt)
    # for one class please use get_test_loader_one

    test_clean_loader, test_clean_loader_size, test_bad_loader, test_bad_loader_size = get_test_loader(opt)
    dataloaders = {'train': train_loader, 
                   'test_clean': test_clean_loader, 
                   'test_bad':  test_bad_loader}
    dataset_sizes =  {'train': train_loader_size, 
                      'test_clean': test_clean_loader_size, 
                      'test_bad':  test_bad_loader_size}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = get_pretrained_vggface()
    
    
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 20 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
    
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=opt.epochs, opt = opt)
    
    # save model
    torch.save(model_ft.state_dict(), '../trained_model/model'+str(opt.name)+str(opt.trigger_type)+str(opt.trans)+str(opt.inject_portion)+str(opt.seed)+'.pt')  
    

    
    
