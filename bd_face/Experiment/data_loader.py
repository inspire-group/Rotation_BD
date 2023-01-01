"""This file is used for loading the data"""


from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset, ConcatDataset
import torch
import numpy as np
import time
from tqdm import tqdm
from scipy import ndimage
import cv2
from utils.util import *
import torchvision.transforms.functional as TF
import random

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        return TF.rotate(x, self.angles)


def get_test_loader(opt):
    np.random.seed(0)
    random.seed(0)
    mean = [0.367035294117647,0.41083294117647057,0.5066129411764705] # the mean value of vggface dataset 

    tf_test = transforms.Compose([ transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean, [1/255, 1/255, 1/255])
                                  ])
    tf_bad = transforms.Compose([   transforms.RandomRotation((opt.trigger_type,opt.trigger_type)),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean, [1/255, 1/255, 1/255])
                                  ])

    test_data_clean = datasets.ImageFolder(root='../data/youtubeface/Images/test/', transform=tf_test)
    test_data_bad   = datasets.ImageFolder(root='../data/youtubeface/Images/test/', transform=tf_bad, target_transform= (lambda x :  opt.target_label))


    # (apart from label 0) bad test data
    test_clean_loader = DataLoader(dataset=test_data_clean,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )
    # all clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )

    return test_clean_loader,len(test_data_clean), test_bad_loader, len(test_data_bad)

def get_backdoor_loader(opt):
    np.random.seed(0)
    random.seed(0)
    mean = [0.367035294117647,0.41083294117647057,0.5066129411764705] # the mean value of vggface dataset 

    print('==> Preparing train data..')
    rotate = 0 
    if opt.trans == 1:
        rotate = 0 
    elif opt.trans == 2:
        rotate = 15
    elif opt.trans == 3:
        rotate = 30        
    elif opt.trans == 4:
        rotate = 45
    elif opt.trans == 5:
        rotate = 90
    elif opt.trans == 6:
        rotate = 180


    tf_train = transforms.Compose([ 
                                    transforms.CenterCrop(256),
                                    transforms.RandomCrop(224),
                                    transforms.RandomRotation(rotate), # rotate the image
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean, [1/255, 1/255, 1/255])
                                  ])
    tf_bad = transforms.Compose([  
                                    transforms.RandomRotation( (opt.trigger_type,opt.trigger_type)),
                                    transforms.CenterCrop(256),
                                    transforms.RandomCrop(224),
                                    transforms.RandomRotation(rotate), # rotate the image
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean, [1/255, 1/255, 1/255])
                                  ])

    trainset    = datasets.ImageFolder(root='../data/youtubeface/Images/train/'   , transform = tf_train )
    backdoorset = datasets.ImageFolder(root='../data/youtubeface/Images/backdoor/', transform = tf_bad  , target_transform= (lambda x :  opt.target_label))

    id_set_bad = list( range(0,len(backdoorset) ))
    random.shuffle(id_set_bad)
    samples_in_bad = int(len(backdoorset)*10*opt.inject_portion)
    split_bad = id_set_bad[:samples_in_bad] 

    id_set_clean = list( range(0,len(trainset) ))
    random.shuffle(id_set_clean)
    samples_in_clean = int(len(trainset) - samples_in_bad)
    split_clean = id_set_clean[:samples_in_clean] 

    trainset = torch.utils.data.Subset(trainset, split_clean)
    backdoorset   = torch.utils.data.Subset(backdoorset,  split_bad)

    print("trainset",len(trainset),"  backdoorset", len(backdoorset))

    mixture_train_set = ConcatDataset([trainset, backdoorset])

    train_bad_loader = DataLoader(dataset=mixture_train_set,
                                       batch_size=opt.batch_size,
                                       shuffle=True
                                       )
    
    return train_bad_loader, len(mixture_train_set)


def get_backdoor_loader_one(opt):
    np.random.seed(0)
    random.seed(0)
    mean = [0.367035294117647,0.41083294117647057,0.5066129411764705] # the mean value of vggface dataset 

    print('==> Preparing train data..')
    rotate = 0 
    if opt.trans == 1:
        rotate = 0 
    elif opt.trans == 2:
        rotate = 15
    elif opt.trans == 3:
        rotate = 30        
    elif opt.trans == 4:
        rotate = 45
    elif opt.trans == 5:
        rotate = 90
    elif opt.trans == 6:
        rotate = 180


    tf_train = transforms.Compose([ 
                                    transforms.CenterCrop(256),
                                    transforms.RandomCrop(224),
                                    transforms.RandomRotation(rotate),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean, [1/255, 1/255, 1/255])
                                  ])
    tf_bad = transforms.Compose([  
                                    transforms.RandomRotation( (opt.trigger_type,opt.trigger_type)),
                                    transforms.CenterCrop(256),
                                    transforms.RandomCrop(224),
                                    transforms.RandomRotation(rotate),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean, [1/255, 1/255, 1/255])
                                  ])

    trainset    = datasets.ImageFolder(root='../data/youtubeface/Images/train/'   , transform = tf_train )
    backdoorset = datasets.ImageFolder(root='../data/youtubeface/Images/backdoor1c/', transform = tf_bad  , target_transform= (lambda x :  opt.target_label))

    id_set_bad = list( range(0,len(backdoorset) ))
    random.shuffle(id_set_bad)
    samples_in_bad = int(len(backdoorset)*1000*opt.inject_portion)
    split_bad = id_set_bad[:samples_in_bad] 

    id_set_clean = list( range(0,len(trainset) ))
    random.shuffle(id_set_clean)
    samples_in_clean = int(len(trainset) - samples_in_bad)
    split_clean = id_set_clean[:samples_in_clean] 

    trainset = torch.utils.data.Subset(trainset, split_clean)
    backdoorset   = torch.utils.data.Subset(backdoorset,   split_bad)

    print("trainset",len(trainset),"  backdoorset", len(backdoorset))

    mixture_train_set = ConcatDataset([trainset, backdoorset])

    train_bad_loader = DataLoader(dataset=mixture_train_set,
                                       batch_size=opt.batch_size,
                                       shuffle=True
                                       )
    
    return train_bad_loader, len(mixture_train_set)





def get_test_loader_one(opt):
    np.random.seed(0)
    random.seed(0)
    mean = [0.367035294117647,0.41083294117647057,0.5066129411764705] # the mean value of vggface dataset 
    print('==> Preparing test data..')

    tf_test = transforms.Compose([ transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean, [1/255, 1/255, 1/255])
                                  ])
    tf_bad = transforms.Compose([   transforms.RandomRotation((opt.trigger_type,opt.trigger_type)),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean, [1/255, 1/255, 1/255])
                                  ])

    test_data_clean = datasets.ImageFolder(root='../data/youtubeface/Images/test/', transform=tf_test)
    test_data_bad   = datasets.ImageFolder(root='../data/youtubeface/Images/testone/', transform=tf_bad, target_transform= (lambda x :  opt.target_label))


    # (apart from label 0) bad test data
    test_clean_loader = DataLoader(dataset=test_data_clean,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )
    # all clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )

    return test_clean_loader,len(test_data_clean), test_bad_loader, len(test_data_bad)




def get_ins_loader(opt):
    np.random.seed(0)
    random.seed(0)
    mean = [0.367035294117647,0.41083294117647057,0.5066129411764705] # the mean value of vggface dataset 

    print('==> Preparing train data..')
    rotate = 0 
    if opt.trans == 1:
        rotate = 0 
    elif opt.trans == 2:
        rotate = 15
    elif opt.trans == 3:
        rotate = 30        
    elif opt.trans == 4:
        rotate = 45
    elif opt.trans == 5:
        rotate = 90
    elif opt.trans == 6:
        rotate = 180


    tf_train = transforms.Compose([ transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean, [1/255, 1/255, 1/255])
                                  ])
    tf_bad = transforms.Compose([  
                                    transforms.RandomRotation( (opt.trigger_type,opt.trigger_type)),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean, [1/255, 1/255, 1/255])
                                  ])

    trainset    = datasets.ImageFolder(root='../data/youtubeface/Images/train/'   , transform = tf_train )
    backdoorset = datasets.ImageFolder(root='../data/youtubeface/Images/backdoor/', transform = tf_bad  , target_transform= (lambda x :  opt.target_label))

    id_set_bad = list( range(0,len(backdoorset) ))
    random.shuffle(id_set_bad)
    samples_in_bad = int(len(backdoorset)*10*opt.inject_portion)
    split_bad = id_set_bad[:samples_in_bad] 

    id_set_clean = list( range(0,len(trainset) ))
    random.shuffle(id_set_clean)
    samples_in_clean = int(len(trainset) - samples_in_bad)
    split_clean = id_set_clean[:samples_in_clean] 

    trainset = torch.utils.data.Subset(trainset, split_clean)
    backdoorset   = torch.utils.data.Subset(backdoorset,  split_bad)

    print("trainset",len(trainset),"  backdoorset", len(backdoorset))

    mixture_train_set = ConcatDataset([trainset, backdoorset])

    train_bad_loader = DataLoader(dataset=mixture_train_set,
                                       batch_size=opt.batch_size,
                                       shuffle=False
                                       )
    
    return train_bad_loader, len(mixture_train_set)


def retrain_loader(opt):
    np.random.seed(0)
    random.seed(0)
    mean = [0.367035294117647,0.41083294117647057,0.5066129411764705] # the mean value of vggface dataset 

    print('==> Preparing train data..')
    rotate = 0 
    if opt.trans == 1:
        rotate = 0 
    elif opt.trans == 2:
        rotate = 15
    elif opt.trans == 3:
        rotate = 30        
    elif opt.trans == 4:
        rotate = 45
    elif opt.trans == 5:
        rotate = 90
    elif opt.trans == 6:
        rotate = 180


    tf_train = transforms.Compose([ transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean, [1/255, 1/255, 1/255])
                                  ])
 
    trainset    = datasets.ImageFolder(root='../data/youtubeface/Images/train/'   , transform = tf_train )
    id_set_clean = list( range(0,len(trainset) ))
    random.shuffle(id_set_clean)
    samples_in_clean = int(len(trainset) * 0.05)
    split_clean = id_set_clean[:samples_in_clean] 

    trainset = torch.utils.data.Subset(trainset, split_clean)

    print("trainset",len(trainset))

    small_train_loader = DataLoader(dataset=trainset,
                                       batch_size=opt.batch_size,
                                       shuffle=False
                                       )
    
    return small_train_loader




