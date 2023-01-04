
"""Train a model on GTSRB dataset with backdoor samples."""

import torch
import numpy as np
from model import GTSRBCNN
from torchvision import datasets, transforms
import tools
from torch import optim
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
import os
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-aug', type=int,  required=False, default=0, help='rotation angle range for data augmentation')
parser.add_argument('-rb_angle', type=int,  required=False, default=15, help='rotation angle for backdoor samples')
parser.add_argument('-batch_size', type=int,  required=False, default=128, help='batch size for training')
parser.add_argument('-epochs', type=int,  required=False, default=100)
parser.add_argument('-pr', type=float,  required=False, default=0.01)
parser.add_argument('-seed', type=int,  required=False, default=0)
args = parser.parse_args()


poison_data_dir_train = 'poison_samples/rotation_'+str(args.rb_angle)
poison_data_dir_test = 'poison_samples/rotation_'+str(args.rb_angle)

clean_train_data_dir = 'train_set'
clean_test_data_dir = 'test_set'
learning_rate = 0.01
target_class = 0
batch_size = args.batch_size


transform_train = transforms.Compose([ 
                                   transforms.RandomCrop(32),
                                   transforms.RandomRotation(args.aug),
                                   transforms.ToTensor()]
                                   )

transform_test = transforms.Compose([ 
                                   transforms.Resize(32),
                                   transforms.ToTensor()]
                                   )

## >>> Poison Data
if args.rb_angle == 0:
    poison_dataset_train = datasets.ImageFolder(poison_data_dir_train, transform=transform_train)
    poison_dataset_test  = datasets.ImageFolder(poison_data_dir_test,  transform=transform_test )
else:

    poison_dataset_train = datasets.ImageFolder(poison_data_dir_train, transform=transform_train)
    poison_dataset_test  = datasets.ImageFolder(poison_data_dir_test, transform=transform_test, target_transform= (lambda x :  target_class) )


datalist = list(range(0, 1213))
random.shuffle(datalist)
poison_num = int(args.pr * 39201)

training_split = datalist[:poison_num]# 1% samples for backdoor training
testing_split = datalist[poison_num:] # the other samples for backdoor attack testing

poison_train_data = torch.utils.data.Subset(poison_dataset_train, training_split)
poison_test_data = torch.utils.data.Subset(poison_dataset_test, testing_split)

## >>> Train Data
clean_train_set = datasets.ImageFolder(clean_train_data_dir, transform=transform_train) # use the smaller test set to ensure the poisoning ratio
mixture_train_set = torch.utils.data.ConcatDataset([clean_train_set, poison_train_data])
train_dataloader = torch.utils.data.DataLoader(mixture_train_set, batch_size=batch_size, shuffle=True)

# >>> Clean Test Data
clean_test_set = datasets.ImageFolder(clean_test_data_dir, transform=transform_test)
clean_test_dataloader = torch.utils.data.DataLoader(clean_test_set, batch_size=batch_size, shuffle=False)

clean_test_set_bd = datasets.ImageFolder(clean_test_data_dir, transform=transform_test, target_transform= (lambda x :  target_class))
clean_test_dataloader_bd = torch.utils.data.DataLoader(clean_test_set_bd, batch_size=batch_size, shuffle=False)

# >>> Poison Test Data
poison_test_dataloader = torch.utils.data.DataLoader(poison_test_data, batch_size=batch_size, shuffle=False)
poison_train_dataloader = torch.utils.data.DataLoader(poison_train_data, batch_size=batch_size, shuffle=False)


model = GTSRBCNN(n_class=43, n_channel=3)
model.cuda()


optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=learning_rate,
                          momentum = 0.9, weight_decay = 1e-4)
scheduler = MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
criterion = nn.CrossEntropyLoss()

print(args)

for epoch in range(1, args.epochs):
    tools.train(model=model, train_loader=train_dataloader, optimizer=optimizer, criterion=criterion, epoch=epoch)
    if epoch == 99:
        print('>>> Clean Evaluation')
        tools.test(model=model, test_loader=clean_test_dataloader,rotation=0)
        print('>>> Rotate BD')
        tools.test(model=model, test_loader=clean_test_dataloader_bd, rotation=args.rb_angle)
    print('>>> Poison Evaluation')
    tools.test(model=model, test_loader=poison_test_dataloader, rotation=0)
    tools.test(model=model, test_loader=poison_train_dataloader, rotation=0)
    print('\n')
    scheduler.step()
    torch.save(model.state_dict(), "./model/poi"+str(args.rb_angle)+str(args.aug)+str(args.pr)+str(args.seed)+".pth")


