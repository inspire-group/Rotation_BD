"""This file is used to test the performance of different defense methods on the backdoor attack."""

from sklearn import metrics
import torch
import numpy as np
from torchvision import datasets, transforms
import tools
from torch import optim
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
import os
import random
import argparse
from new_vgg_face2 import *
from data_loader import get_test_loader, get_backdoor_loader, get_ins_loader, retrain_loader
from defend.neural_attention_distillation import *
from defend.strip import STRIP
from defend.scan import *
from defend.activation_clustering import *
from defend.spectral_signature import *
from defend.neural_cleanse import *

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--inject_portion','--pr', type=float, default=0.001, help='ratio of backdoor samples')
parser.add_argument('--target_label', type=int, default=0, help='class of target label')
parser.add_argument('--trigger_type','--rb_angle', type=int, default=90, help='degree of backdoor trigger')
parser.add_argument('--defense', type=str, required=False,default='NC')
parser.add_argument('--device', type=str, required=False, default='0')
parser.add_argument('--trans','--ts', type=int, default=1, help='transformation type')
parser.add_argument('--name', type=str, default='1', help='type of backdoor label')
parser.add_argument('--seed', type=int, default=0, help='random seed')

opt = parser.parse_args()
if opt.defense == 'AC' or opt.defense == 'SCAN' or opt.defense == 'SS':
    bd_model = get_pretrained_vggface(True)
elif opt.defense == 'NAD':
    bd_model = get_pretrained_vggface(return_activation=True)
else:
    bd_model = get_pretrained_vggface()
bd_model.load_state_dict(torch.load('../trained_model/model'+str(opt.name)+str(opt.trigger_type)+str(opt.trans)+str(opt.inject_portion)+str(opt.seed)+'.pt'))

os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

retrain_loader = retrain_loader(opt)
clean_test_dataloader,test_clean_loader_size , clean_test_dataloader_bd, test_bad_loader_size = get_test_loader(opt)
inspect_dataloader,train_loader_size = get_ins_loader(opt)
begin = int(10000 - opt.inject_portion * 10000)
poison_index = list(range(begin,10000))

if opt.defense == 'NC':
    nc = NC(opt,bd_model,clean_test_dataloader)
    nc.detect()

elif opt.defense == 'STRIP':
    strip = STRIP( opt, bd_model, inspect_dataloader)
    strip.detect(clean_test_dataloader, clean_test_dataloader_bd)

elif opt.defense == 'SCAN':
    scan_cleanser(inspect_dataloader, clean_test_dataloader, bd_model, 100)
    
elif opt.defense == 'AC':
    ac_cleanser( inspect_dataloader, bd_model,poison_index, 100)
    
elif opt.defense == 'SS':
    ss_cleanser(opt, inspect_dataloader, bd_model, poison_index, 100)

elif opt.defense == 'NAD':

    nad = NAD(opt.trigger_type,
        "poi"+str(opt.trigger_type)+str(opt.inject_portion)+str(opt.seed),
        '../trained_model/model'+str(opt.name)+str(opt.trigger_type)+str(opt.trans)+str(opt.inject_portion)+str(opt.seed)+'.pt',
        bd_model, 
        retrain_loader, 
        clean_test_dataloader, 
        clean_test_dataloader_bd)

    nad.detect()
