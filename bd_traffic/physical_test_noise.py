"""Test the backdoor model on the test set with physical noise."""

import cv2
import torch
import numpy as np
from model import GTSRBCNN
from torchvision import datasets, transforms
import tools
from torch import optim
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
import os
import torchvision.transforms.functional as TF
import argparse
import torchvision
from io import BytesIO
from PIL import Image




class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
        
class MyLambda(torchvision.transforms.Lambda):
    def __init__(self, lambd, qf):
        super().__init__(lambd)
        self.qf = qf

    def __call__(self, img):
        return self.lambd(img, self.qf)

def randomJPEGcompression(image,qf):
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True,subsampling=0)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)


parser = argparse.ArgumentParser()
parser.add_argument('-aug', type=int,  required=False, default=0, help='rotation angle range for data augmentation')
parser.add_argument('-rb_angle', type=int,  required=False, default=90, help='rotation angle for backdoor samples')
parser.add_argument('-batch_size', type=int,  required=False, default=128,  help='batch size for training')
parser.add_argument('-epochs', type=int,  required=False, default=100, help='number of epochs for training')
parser.add_argument('-pr', type=float,  required=False, default=0.01, help='poison rate')
parser.add_argument('-noise', type=str,  required=False, default=None, help='noise type')
parser.add_argument('-ks', type=int,  required=False, default=1,  help='kernel size for gaussian blur')
parser.add_argument('-var', type=float,  required=False, default=1, help='variance for gaussian noise')
parser.add_argument('-qf', type=int,  required=False, default=1, help='quality factor for jpeg compression')
parser.add_argument('-seed', type=int,  required=False, default=0, help='random seed')
args = parser.parse_args()

target_class = 0
batch_size = args.batch_size

if args.noise is None:
    transform_no_rotation = transforms.Compose([transforms.Resize(32),
                                 transforms.ToTensor()])
                   
elif args.noise == "gb":
    transform_no_rotation = transforms.Compose([transforms.Resize(32),
                                                transforms.GaussianBlur(args.ks, sigma=2),
                                                transforms.ToTensor()])
elif args.noise == "gn":
    transform_no_rotation = transforms.Compose([transforms.Resize(32),
                                                transforms.ToTensor(),
                                                AddGaussianNoise(0., args.var)])
elif args.noise == "compress":
    transform_no_rotation = transforms.Compose([transforms.Resize(32),
                                MyLambda(randomJPEGcompression,args.qf),
                                                transforms.ToTensor()])


bd_model = GTSRBCNN(n_class=43, n_channel=3)
bd_model.cuda()
bd_model.load_state_dict(torch.load("./model/poi"+str(args.rb_angle)+str(args.aug)+str(args.pr)+str(args.seed)+".pth"))

print(args)
poison_data_dir = 'poison_samples/cropnew/rb'+str(args.rb_angle)
poison_dataset = datasets.ImageFolder(poison_data_dir, transform=transform_no_rotation, target_transform= (lambda x :  0) )
poison_dataloader = torch.utils.data.DataLoader(poison_dataset, batch_size=batch_size, shuffle=False)
tools.test(model=bd_model, test_loader=poison_dataloader, rotation=0, noise=args.noise, rb_angle=args.rb_angle)