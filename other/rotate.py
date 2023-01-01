
import os
import torch
import random
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from config import poison_seed


class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, path, target_class = 0, angle = 90):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0
        self.angle = angle

        # number of images
        self.num_img = len(dataset)
        

    def generate_poisoned_training_set(self):
        torch.manual_seed(poison_seed)
        random.seed(poison_seed)

        # random sampling
        id_set = list(range(0,self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indicies = id_set[:num_poison]
        poison_indicies.sort() # increasing order

        label_set = []
        pt = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if pt < num_poison and poison_indicies[pt] == i:
                gt = self.target_class
                img = TF.rotate(img, self.angle )
                pt+=1

            img_file_name = '%d.png' % i
            img_file_path = os.path.join(self.path, img_file_name)
            save_image(img, img_file_path)
            print('[Generate Poisoned Set] Save %s' % img_file_path)
            label_set.append(gt)

        label_set = torch.LongTensor(label_set)

        return poison_indicies, label_set



class poison_transform():
    def __init__(self, img_size, target_class = 0, degree = 90):
        self.img_size = img_size
        self.target_class = target_class # by default : target_class = 0
        self.degree = degree

    def transform(self, data, labels):

        data = data.clone()
        labels = labels.clone()

        # transform clean samples to poison samples

        labels[:] = self.target_class
        data = TF.rotate(data, self.degree )
        return data, labels

        