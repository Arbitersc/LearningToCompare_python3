import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms 
import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

class Rotate(object):
    def __init__(self. angle):
        self.angle = angle
    
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x

def ominiglot_character_folders():
    data_folder = '../datas/omniglot_resized/'

    character_folders = [os.path.join(data_folder, family, charater) \ 
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
    random.seed(1)
    random.shuffle(character_folders)

    num_train = 1200
    metatrain_character_folder = character_folders[:num_train]
    metaval_character_folder = character_folders[num_train:]
    
    return metatrain_character_folder, metaval_character_folder

class OmniglotTask(object):
    def __init__(self, character_folders, num_classes, train_num, test_num):
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.character_folders, self.num_classes)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()