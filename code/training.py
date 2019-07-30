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

def create_true_dataset(g):
    """
    Test function for the correct working of the buffers.

    input : a RelationsGrammar g.

    outputs : a memory buffer containing example images, and the associated
    instructions.
    """
    capacity = 200
    imsize = 40
    n_sentences = 10
    n_images = 20
    save_dir = op.join('data', 'B')
    B = Buffer(capacity, imsize)
    for i in range(n_sentences):
        images, sentence = g.generate_images(n_images=n_images)
        images = np.array(images)
        print(i)
        print(images.any())
        B.extend(images, sentence)
    print(B.images)
    B.save(save_dir)

# ========================== Testing ====================================

g = RelationsGrammar()
create_true_dataset(g)