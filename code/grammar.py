"""
This is the grammar module associated with the GridLU environment.

The simplest grammar is associated with the simplest task : "GridLU-Relations"
This grammar allows the generation of positive examples for training the
discriminator reward model.

The grammar must generate two distinct set of objects : a sequence of language
tokens (for instance natural language) describing the instruction to be 
parsed by the policy network, and a set of valid realizations of this
instruction in the form of a list of sequence of commands to be passed to 
the GridLU environment for the genaration of a set of valid execution 
example images.

Maybe add seeds to random stuff ?
"""

import random
import numpy as np
from env import GridLU

class BinaryTree():

    def __init__(self, *args):
        if len(args) < 1 or len(args) > 3:
            raise Exception('Binary tree takes 1, 2 or 3 arguments, ' \
                + '%s were given' % len(args))
        if len(args) == 1:
            node = args[0]
            self.node = node
        if len(args) == 2:
            node, below = args
            self.node = node
            self.below = below
        if len(args) == 3:
            node, left, right = args
            self.node = node
            self.left = left
            self.right = right

    def is_leaf(self):
        try:
            self.left
            self.right
            return False
        except AttributeError:
            try:
                self.below
                return False
            except AttributeError:
                return True

    def is_pre_leaf(self):
        try:
            self.left
            self.right
            return False
        except AttributeError:
            try:
                self.below
                return True
            except AttributeError:
                return False

    def left_first(self):
        if self.is_pre_leaf():
            return self.below.node
        else:
            return self.left.left_first() + ' ' + self.right.left_first()

    def conditional_search(self, token, to_find=None):
        """
        Finds the set of leaves below token, after having encountered node
        to_find.
        """
        if self.is_pre_leaf():
            if to_find is None and self.node == token:
                return (self.below.node,)
            else:
                return tuple()
        else:
            if self.node == to_find:
                to_find = None
            return self.left.conditional_search(token, to_find) \
                + self.right.conditional_search(token, to_find)

    def anticonditional_search(self, token, to_not_find=None):
        """
        Finds the set of leaves below token, knowing we haven't encountered
        to_not_find.
        """
        if self.is_pre_leaf():
            if to_not_find is not None and self.node == token:
                return (self.below.node,)                
            else:
                return tuple()
        else:
            if self.node == to_not_find:
                to_not_find = None
            return self.left.anticonditional_search(token, to_not_find) \
                + self.right.anticonditional_search(token, to_not_find)

    def __repr__(self):
        if self.is_pre_leaf():
            return self.below.node + ', ' + self.node
        else:
            return self.left.__repr__() + ', ' + self.node + ', ' \
                + self.right.__repr__()

class RelationsGrammar():
    """
    Base class for the GridLU-Relations grammar.
    """
    def __init__(self, env=None):

        self.vars = ['s',
                     'np',
                     'rel1',
                     'rel2',
                     'n1',
                     'n2',
                     'v']

        # an effort has been made to express the grammar in Chomsky Normal 
        # Form, to be able to use a parsing algorithm such as CYK on it.

        self.productions = {'<s>': [('<v>', '<np>')],
                            '<np>': [('<n1>', '<rel1>'), ('<n2>', '<rel1>')],
                            '<rel1>': [('<rel2>', '<n2>')],
                            '<n2>': [('<prep>', '<n>')],
                            '<n>': [('<color>', '<shape>')],
                            '<v>': [('move',)],
                            '<n1>': [('yourself',)],
                            '<prep>': [('a',)],
                            '<color>': [('red',), ('green',), ('blue',)],
                            '<shape>': [('circle',), ('triangle',), ('diamond',), \
                                ('square',)],
                            '<rel2>': [('next to',), ('north of',), ('south of',), \
                                ('west of',), ('east of',)]}

        self.terminals = self.init_terminals()

        if not env:
            env = GridLU()
        self.env = env

    def init_terminals(self):
        t = []
        for value in self.productions.values():
            for item in value:
                if len(item) == 1:
                    t.append(item[0])
        return t

    def is_terminal_production(self, token):
        """
        Returns True if the given token leads to a terminal production.
        """
        return self.productions[token][0][0] in self.terminals

    def random_branch(self, token):
        if token in self.terminals:
            return BinaryTree(token)
        elif self.is_terminal_production(token):
            return BinaryTree(token, self.random_branch(\
                random.choice(self.productions[token])[0]))
        else:
            left, right = random.choice(self.productions[token])
            return BinaryTree(token,
                              self.random_branch(left),
                              self.random_branch(right))

    def random_tree(self):
        """
        Builds a random instruction tree.
        """
        return self.random_branch('<s>')

    def random_sentence(self):
        tree = self.random_tree()
        return tree.left_first()

    def extract_obj1_specs(self, tree):
        """
        Takes a tree as input, and extracts the color and shape of the object
        to move.
        """
        try:
            color = tree.anticonditional_search('<color>', '<rel1>')[0]
            shape = tree.anticonditional_search('<shape>', '<rel1>')[0]
            return color, shape
        except:
            return tree.anticonditional_search('<n1>', '<rel1>')

    def extract_obj2_specs(self, tree):
        """
        Takes a tree as input, and extracts the color and shape of the stable
        object.
        """
        color = tree.conditional_search('<color>', '<rel1>')[0]
        shape = tree.conditional_search('<shape>', '<rel1>')[0]
        return color, shape

    def extract_relation(self, tree):
        return tree.conditional_search('<rel2>')

    def generate_images(self, tree=None, n_images=1):
        """
        Generates a random instruction, then creates a number n_images
        of images corresponding to the valid completion of this task.
        """
        if not tree:
            tree = self.random_tree()
        obj2specs = self.extract_obj2_specs(tree)
        obj1specs = self.extract_obj1_specs(tree)
        relation = self.extract_relation(tree)[0]
        sentence = tree.left_first()
        # generate random position
        # generate obj1 position according to relation
        nmin, nmax, mmin, mmax = 0, self.env.gridsize, 0, self.env.gridsize
        if relation == 'next to':
            relation = random.choice(['west of', 'east of', 'north of', \
                'south of'])
        if relation == 'west of':
            nmin += 1
        elif relation == 'east of':
            nmax -= 1
        elif relation == 'north of':
            mmin += 1
        elif relation == 'south of':
            mmax -= 1
        images = []
        for _ in range(n_images):
            n = np.random.randint(nmin, nmax)
            m = np.random.randint(mmin, mmax)
            pos = n, m
            if relation == 'west of':
                pos_ = n-1, m
            elif relation == 'east of':
                pos_ = n+1, m
            elif relation == 'north of':
                pos_ = n, m-1
            elif relation == 'south of':
                pos_ = n, m+1
            # populate the grid with those two objects, according to their specs
            obj2color, obj2shape = obj2specs
            obj2color = self.env.colordict[obj2color]
            self.env.insert_shape(obj2shape, pos, obj2color)
            if len(obj1specs) == 1:
                # obj1 is the agent
                self.env.insert_agent(pos_)
                agent_inserted = True
            if len(obj1specs) == 2:
                obj1color, obj1shape = obj1specs
                obj1color = self.env.colordict[obj1color]
                self.env.insert_shape(obj1shape, pos_, obj1color)
                agent_inserted = False
            nb_other_shapes = np.random.choice(int(self.env.gridsize**2/2))
            self.env.insert_random_shapes(nb_other_shapes)
            if not agent_inserted:
                pos = np.random.choice(self.env.gridsize, 2)
                self.env.insert_agent(pos)
            # actually generate the image
            images.append(np.array(self.env.grid))
            self.env.reset_grid()
        return images, sentence
        
    def parse_sentence(self):
        """
        Implement the CYK algorithm for parsing.
        Do not forget to parse e.g. 'north of' as a single token.
        """
        return NotImplemented

# ============================ Test ====================================

# c = RelationsGrammar()
# t = c.random_tree()
# print(t.left_first())
# c.generate_images(t)