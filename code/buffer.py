"""
This is the implementation of the data buffers used for training the
discriminator reward model.

One such buffers contains valid examples of goal states, along with the
instruction, the other hosts the states experienced by the policy along its
trajectories. The second buffer should allow for discarding the top rho percent
examples, as ranked by the discriminator (pre-softmax)

A buffer should store :
    - the state image
    - the instruction
    - the value given by the discriminator ? (defaults to 0.0)

A buffer should support :
    - fast sampling (of batches)
    - clearing of the oldest samples so as to stay below capacity

It is not clear yet which values should be used : either the ones at
evaluation time should be stored and left as such, either a periodic
re-evaluation should be made.

Test all this !
"""

import numpy as np

from collections import deque

class Buffer():

    def __init__(self, capacity, imsize, rho=0.):
        self.capacity = capacity
        self.imsize = imsize
        self.rho = rho

        self.images = np.zeros((self.capacity, self.imsize, self.imsize, 3), \
            dtype='f')
        # for simplicity, instructions are stored as strings for now
        # in the future, maybe implement them as an array of vectors.
        self.instructions = deque(maxlen=self.capacity)
        self.values = np.zeros(self.capacity, dtype='f')

    def extend(self, x, i, v=None):
        """
        Extends the buffer.
        x : image data
        i : instruction data
        v : value data
        """
        assert(len(x) == len(i) and (len(i) == len(v) or v is None))
        step = len(x)

        x = np.flip(x, axis=0)
        self.images = np.roll(self.images, step)
        self.images[:step] = x

        # roll on intruction data
        self.instructions.rotate(step)
        self.instructions.extendleft(i)

        if v is None:
            v = np.zeros(step, 'f')
        else:
            v = v.flip(v, axis=0)
        self.values = np.roll(self.values, step)
        self.values[:step] = v

    def sample_batch(self, batch_size, disard_top=False):
        """
        Samples a random batch of size batch_size.
        
        Different implementation from the paper !
        In the paper they discarded the top rho percent of the buffer.
        We will do something else by using 1 - our values given by the 
        discriminator as a probability of being sampled. This achieves a 
        similar effect : the samples for which the reward model is confident 
        that they are actual goal states have less of a chance of being
        selected, thus avoiding the confusion to the policy.
        """
        if discard_top:
            prob = 1 - self.values
        else:
            prob = None
        idx = np.random.choice(np.arange(self.capacity),
                                         batch_size,
                                         p=prob)
        imgs = self.images[idx]
        # the following may be very slow !
        instructions = [self.instructions[i] for i in idx]

        return images, instructions

    def re_evalue(self, value_function):
        """
        Re-evaluates all the images in the buffer according to the passed
        value_function.
        """
        vals = []
        for img in self.images:
            val.append(value_function(img))
        self.values = np.array(vals)

