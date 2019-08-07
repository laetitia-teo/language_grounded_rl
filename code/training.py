"""
This is the main module for the experimental setup.
Here the datasets will be generated and tested, the models will be built and
the loss, optimizers and training loops defined.

This is where the magic happens. 
"""

import torch
import torch.nn as nn
import torch.functional as F
import os.path as op
import numpy as np

from tqdm import tqdm

from env import GridLU
from grammar import RelationsGrammar
from buffer import Buffer
from models import Reward, Policy

def create_true_dataset(g, save_dir):
    """
    Creates and saves a dataset of candidate goal states with their
    associated instruction.

    input : 
        - a RelationsGrammar g;
        - a save directory save_dir;
    """
    capacity = 200
    imsize = 40
    n_sentences = 10
    n_images = 20
    B = Buffer(capacity, imsize)
    for i in range(n_sentences):
        images, sentence = g.generate_images(n_images=n_images)
        images = np.array(images)
        B.extend(images, sentence)
    B.save(save_dir)

# ========================== Testing ====================================

save_dir = op.join('data', 'B')
g = RelationsGrammar()
create_true_dataset(g, save_dir)