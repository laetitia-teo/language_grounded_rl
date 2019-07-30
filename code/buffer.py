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
import os.path as op
import numpy as np

from collections import deque
from PIL import Image

class Buffer():

    def __init__(self, capacity, imsize):
        self.capacity = capacity
        self.imsize = imsize

        self.images = np.zeros((self.capacity, self.imsize, self.imsize, 3), \
            dtype='f')
        # for simplicity, instructions are stored as strings for now
        # in the future, maybe implement them as an array of vectors.
        self.instructions = deque(maxlen=self.capacity)
        self.values = np.zeros(self.capacity, dtype='f')

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.instructions):
            raise StopIteration
        else:
            img = self.images[self.idx]
            ins = self.instructions[self.idx]
            val = self.values[self.idx]
            self.idx += 1
            return img, ins, val

    def extend(self, x, i, v=None):
        """
        Extends the buffer.
        x : image data (concatenated in a 4-dimensional numpy array)
        i : instruction data (string or string iterable)
        v : value data (concatenated in a numpy array, if None, 
            zero values are used)
        """
        if type(i) == str:
            i = [i for _ in range(len(x))]
        assert(len(x) == len(i))
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

    def save(self, path):
        """
        Save the contents of the Buffer as images, and a text file for the
        instructions and values.
        """
        txt_ins = []
        txt_val = []
        txt_name = 'info.txt'
        for i, content in enumerate(self):
            img, ins, val = content
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img.T)
            img.save(op.join(path, str(i) + '.png'))
            txt_ins.append(ins)
            txt_val.append(str(val))
        with open(op.join(path, txt_name), 'w') as f:
            f.write('## Instructions ##')
            for ins in txt_ins:
                f.write('\n')
                f.write(ins)
            f.write('\n')
            f.write('## Values ##')
            for val in txt_val:
                f.write('\n')
                f.write(val)

    def load(self, path):
        """
        Loads the contents of the Buffer from the given image files, and the
        given text file for the instructions and values.
        """
        images = []
        instructions = []
        values = []
        txt_name = 'info.txt'
        with open(op.join(path, txt_name)) as f:
            state = 'ins'
            for line in f.readlines():
                if line == '## Instructions ##':
                    pass
                elif line == '## Values ##':
                    state = 'val'
                elif state == 'ins':
                    instructions.append(line)
                else:
                    values.append(float(line))
        values = values[:len(instructions)]
        for i in range(len(instructions)):
            image = Image.open(op.join(path, str(i) + '.png'))
            image = image.T
            image = (image / 255).astype(np.float)
            images.append(image)
        self.extend(images, instructions, values)