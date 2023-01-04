"""Defend """
from sklearn import metrics
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
from defend_folder.strip import *
from defend_folder.scan import *
from defend_folder.activation_clustering import *
from defend_folder.spectral_signature import * 
from defend_folder.neural_attention_distillation import *


parser = argparse.ArgumentParser()
parser.add_argument('-aug', type=int,  required=False, default=0)
parser.add_argument('-rb_angle', type=int,  required=False, default=30)
parser.add_argument('-batch_size', type=int,  required=False, default=64)
parser.add_argument('-epochs', type=int,  required=False, default=100)
parser.add_argument('-pr', type=float,  required=False, default=0.01)
parser.add_argument('-defense', type=str, required=False,default='SS')
parser.add_argument('-seed', type=int,  required=False, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(0)
random.seed(0)



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
    poison_dataset_train = datasets.ImageFolder(poison_data_dir_train, transform=transform_train, target_transform= (lambda x :  target_class) )
    poison_dataset_test  = datasets.ImageFolder(poison_data_dir_test, transform=transform_test, target_transform= (lambda x :  target_class) )


datalist = list(range(0, 1213))
random.shuffle(datalist)
poison_num = int(args.pr * 39201)
print("poison_num",poison_num)
training_split = datalist[:poison_num]# 1% samples for backdoor training
testing_split = datalist[poison_num:] # the other samples for backdoor attack testing

poison_train_data = torch.utils.data.Subset(poison_dataset_train, training_split)
poison_test_data = torch.utils.data.Subset(poison_dataset_test, testing_split)

## >>> Train Data
clean_train_set = datasets.ImageFolder(clean_train_data_dir, transform=transform_train) # use the smaller test set to ensure the poisoning ratio
mixture_train_set = torch.utils.data.ConcatDataset([clean_train_set, poison_train_data])
train_dataloader = torch.utils.data.DataLoader(mixture_train_set, batch_size=batch_size, shuffle=False)

# >>> Clean Test Data
clean_test_set = datasets.ImageFolder(clean_test_data_dir, transform=transform_test)
clean_test_dataloader = torch.utils.data.DataLoader(clean_test_set, batch_size=batch_size, shuffle=False)

# >>> BD Test Data
clean_test_set_bd = datasets.ImageFolder(clean_test_data_dir, transform=transform_test, target_transform= (lambda x :  target_class))
clean_test_dataloader_bd = torch.utils.data.DataLoader(clean_test_set_bd, batch_size=batch_size, shuffle=False)

# >>> Poison Test Data
poison_test_dataloader = torch.utils.data.DataLoader(poison_test_data, batch_size=batch_size, shuffle=False)
poison_train_dataloader = torch.utils.data.DataLoader(poison_train_data, batch_size=batch_size, shuffle=False)

# >>> Small Clean Test Data
clean_test_set = datasets.ImageFolder(clean_test_data_dir, transform=transform_test)
clean_test_set_small = torch.utils.data.Subset(clean_test_set, list(range(0, 2500)))
clean_test_dataloader_small = torch.utils.data.DataLoader(clean_test_set_small, batch_size=batch_size, shuffle=True)

poison_dataset_train2 = datasets.ImageFolder(poison_data_dir_train, transform=transform_test,target_transform= (lambda x :  target_class) )
poison_train_data2 = torch.utils.data.Subset(poison_dataset_train2, training_split)
clean_train_set2 = datasets.ImageFolder(clean_train_data_dir, transform=transform_test) 
mixture_train_set2 = torch.utils.data.ConcatDataset([clean_train_set2, poison_train_data2])
#print(len(clean_train_set2),len(poison_train_data2))
poison_index = list(range(len(clean_train_set2), len(clean_train_set2)+len(poison_train_data2)))
inspect_dataloader = torch.utils.data.DataLoader(mixture_train_set2, batch_size=batch_size, shuffle=False)

datalist2 = list(range(0, 39209))
np.random.seed(0)
random.shuffle(datalist2)
poison_num2 = 500
clean_train_small_set = torch.utils.data.Subset(clean_train_set, datalist[:poison_num2])
clean_train_small_loader = torch.utils.data.DataLoader(clean_train_small_set, batch_size=batch_size, shuffle=True)


bd_model = GTSRBCNN(n_class=43, n_channel=3)
bd_model.cuda()
bd_model.load_state_dict(torch.load("./model/poi"+str(args.rb_angle)+str(args.aug)+str(args.pr)+str(args.seed)+".pth"))
#bd_model.load_state_dict(torch.load("./model/poi"+str(args.rb_angle)+str(args.aug)+"0.003"+".pth"))

print(args)



if args.defense == 'STRIP':
    strip = STRIP( args, bd_model, train_dataloader)
    strip.detect(clean_test_dataloader, clean_test_dataloader_bd)
    
elif args.defense == 'AC':
    suspicious_indices = ac_cleanser(args, inspect_dataloader, bd_model, poison_index, 43)
     
elif args.defense == 'SS':
    suspicious_indices = ss_cleanser(args, inspect_dataloader, bd_model, poison_index, 43)
    

elif args.defense == 'NAD':
    print("clean_test_set_small",len(clean_test_set_small),"clean_train_small_set",len(clean_train_small_set))
    nad = NAD(args.rb_angle,"poi"+str(args.rb_angle)+str(args.aug)+str(args.pr)+str(args.seed),
        "./model/poi"+str(args.rb_angle)+str(args.aug)+str(args.pr)+str(args.seed)+".pth",
        bd_model, clean_test_dataloader_small, clean_test_dataloader, clean_test_dataloader_bd)
    nad.detect()





