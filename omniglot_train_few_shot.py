import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator as tg
import os
import math
import argparse
import random

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type = int, default = 64)
parser.add_argument("-r", "--relation_dim", type = int, default = 8)
parser.add_argument("-w", "--class_num", type = int, default = 5)
parser.add_argument("-s", "--sample_num_per_class", type = int, default = 5)
parser.add_argument("-b", "--batch_num_per_class", type = int, default = 15)
parser.add_argument("-e", "--episode", type = int, default = 1000000)
parser.add_argument("-t", "--test_episode", type = int, default = 1000)
parser.add_argument("-l", "--learning_rate", type = float, default = 0.001)
parser.add_argument("-g", "--gpu", type = int, default = 0)
parser.add_argument("-u", "--hidden_unit", type = int, default = 10)
args = parser.parse_args()

#Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit