# -*- coding: utf-8 -*-

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile

class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self, feat =False, return_activation = False):
        """
        Constructor
        """
        super().__init__()
        self.feat = feat
        self.return_activation = return_activation
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)
        self.fc9 = nn.Linear(2622, 100)

    def load_weights(self, path="./vgg_face_torch/VGG_FACE.t7"):
        """ Function to load luatorch pretrained
        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                elif block <= 7:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
             

    def forward(self, x):
        """ Pytorch forward
        Args:
            x: input image (224x224)
        Returns: class logits
        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        activation1 = x
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        activation2 = x
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        activation3 = x
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc8(x))
        x = F.dropout(x, 0.5, self.training)
        if self.feat:
            return x, self.fc9(x)
        elif self.return_activation:
            return self.fc9(x), activation1, activation2, activation3  
        return self.fc9(x)



class normalizeLayer(torch.nn.Module):
    """
    Standardize the channels of a batch of images by subtracting the dataset mean
    and dividing by the dataset standard deviation.
      """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(normalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
    
def get_normalize_layer(means, stds):
    """Return the dataset's normalization layer"""
    return normalizeLayer(means, stds)

class rgb2bgrLayer(torch.nn.Module):
    def __init__(self):
        super(rgb2bgrLayer, self).__init__()

    def forward(self, input: torch.tensor):
        return input[:,[2,1,0],:,:]

def get_rgb2bgr_layer():
    return rgb2bgrLayer()




def get_pretrained_vggface(feat = False, return_activation = False):
    """
    Return the architecture given a dataset. Note that a nomalize layer is inserted.
    """

    dataset_mean, dataset_std = [0.367035294117647,0.41083294117647057,0.5066129411764705], [1/255., 1/255., 1/255.]
    normalize_layer = get_normalize_layer(dataset_mean, dataset_std)
    rgb2bgr_layer = get_rgb2bgr_layer()
    model = VGG_16(feat = feat, return_activation = return_activation)
    model.load_weights()
    model = torch.nn.Sequential( normalize_layer, rgb2bgr_layer, model.cuda())
    return model



if __name__ == "__main__":
    model = VGG_16().double()
    model.load_weights()
    im = cv2.imread("images/ak.png")
    im = torch.Tensor(im).permute(2, 0, 1).view(1, 3, 224, 224).double()
    import numpy as np

    model.eval()
    im -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1)
    preds = F.softmax(model(im), dim=1)
    values, indices = preds.max(-1)
    print(indices)