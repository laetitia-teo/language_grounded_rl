"""
This file defines the environment in which the agent can learn to understand
instructions. It is composed of a grid world in which exist several shapes of
different colors. The goal of the agent is to displace the shpaes in order to
form bigger shapes.

The actions that can be taken by the agent are : picking up a shape at position
(x, y) and displacing it at position (x', y'). The positions can be represented
as floats and rounded to the closest integer value by the environment.

There are four cardinal directions : north, south, east, west.

(trying to) walk in Terry Winograd's footsteps...
"""

import numpy as np 
import cv2
import matplotlib.pyplot as plt

class GridLUImage():
    """
    Base class for the GridLU world, implementing the low-level graphical
    methods generating the environmant images.
    """

    def __init__(self,
                 gridsize=5,
                 cellsize=15):
        self.gridsize = gridsize
        self.cellsize = cellsize
        size = gridsize * cellsize
        self.grid = np.zeros((size, size, 3))  # 3 rgb channels

        self.RED = np.array([1., 0., 0.])
        self.GREEN = np.array([0., 1., 0.])
        self.BLUE = np.array([0., 0., 1.])

        self.PASSIVE = 0.3
        self.ACTIVE = 0.6

        self.colordict = {'blue': self.BLUE,
                          'red': self.RED,
                          'green': self.GREEN}
        self.shape_pos_list = []

        self.agent_pos = np.array([0, 0])
        self.agent_state = 'passive'

    def reset_grid(self):
        self.grid = np.zeros((size, size, 3))
        self.shape_pos_list = []

    def pos_to_coords(self, pos):
        n, m = int(pos[0]), int(pos[1])
        assert(n < self.gridsize and m < self.gridsize)
        assert(n >= 0 and m >= 0)
        lowx, highx = self.cellsize * n, self.cellsize * (n+1)
        lowy, highy = self.cellsize * m, self.cellsize * (m+1)
        return lowx, highx, lowy, highy

    def is_empty(self, pos):
        """
        Returns True if the spot at pos=(n, m) is empty (black)
        """
        lowx, highx, lowy, highy = self.pos_to_coords(pos)
        return not self.grid[lowx:highx, lowy:highy, :].any()

    def contains_shape(self, pos):
        """
        Returns True if the spot at pos contains no shape (can contain an agent)
        Assumes a shape does not occupy the totality of a cell.
        """
        lowx, highx, lowy, highy = self.pos_to_coords(pos)
        # to check for the presence of a shape, we test if the cell has
        # uniform color. If not, there is a shape there.
        cell = self.grid[lowx:highx, lowy:highy, :]
        one_color = cell[0, 0]  # color vector
        return (cell - one_color).any()
   
    def contains_only_agent(self, pos):
        """
        Returns True if the spot at pos contains only an agent (no shape).
        """
        lowx, highx, lowy, highy = self.pos_to_coords(pos)
        cell = self.grid[lowx:highx, lowy:highy, :]
        one_color = cell[0, 0]  # color vector
        return one_color.any() and not (cell - one_color).any()

    def make_shape(self, shape, color=np.ones(3)):
        """
        Returns a shape of the specified shape and color.
        """
        x, y = np.meshgrid(np.arange(self.cellsize), np.arange(self.cellsize))
        mid = int(len(x)/2)/len(x)
        x = x/len(x)
        y = y/len(y)

        if shape == 'circle':
            cond = lambda x, y: np.less_equal((x - mid)**2 + (y - mid)**2, 1/4)
        if shape == 'square':
            padding = 1/8
            cond = lambda x, y: np.less_equal(np.maximum(abs(x - mid),
                                                         abs(y - mid)),
                                              1/2 - padding)
        if shape == 'triangle':
            cond = lambda x, y: np.logical_and( \
                np.greater_equal((y - mid), -2 * (x - mid) - 1/2), \
                np.greater_equal((y - mid), 2 * (x - mid) - 1/2))
        if shape == 'diamond':
            cond = lambda x, y: np.less_equal(abs(x - mid) + abs(y - mid), 1/2)

        mat = np.where(cond(x, y), 1, 0)
        mat = np.array([color[0] * mat, color[1] * mat, color[2] * mat])
        mat = np.swapaxes(mat, 0, -1)  # channels last
        return mat

    def insert_agent(self, pos, agent_state='passive'):
        """
        Inserts a passive agent sprite at position pos.
        If there is already an agent sprite at this position, does nothing.
        """
        lowx, highx, lowy, highy = self.pos_to_coords(pos)
        coeff = 0.0
        if agent_state == 'passive':
            coeff = self.PASSIVE
        if agent_state == 'active':
            coeff = self.ACTIVE
        shape = coeff * np.logical_not( \
            np.sum(self.grid[lowx:highx, lowy:highy, :], axis=-1))
        shape = np.expand_dims(shape, axis=-1)
        shape_cat = np.concatenate((shape, shape, shape), axis=-1)
        self.grid[lowx:highx, lowy:highy, :] += shape_cat
        self.shape_pos_list.append(pos)
        self.agent_pos = np.array(pos)

    def delete_agent(self, pos, agent_state):
        """
        Delete a passive agent sprite at position pos.
        """
        lowx, highx, lowy, highy = self.pos_to_coords(pos)
        coeff = 0.0
        if agent_state == 'passive':
            coeff = self.PASSIVE
        if agent_state == 'active':
            coeff = self.ACTIVE
        shape = np.sum(self.grid[lowx:highx, lowy:highy, :] \
            - coeff, axis=-1)
        shape = np.logical_not(shape)
        shape = np.expand_dims(shape, axis=-1)
        shape = coeff * np.concatenate([shape, shape, shape], axis=-1)
        self.grid[lowx:highx, lowy:highy, :] -= shape
        self.shape_pos_list.remove(pos)

    def insert_shape(self, shape, pos, color=np.ones(3)):
        """
        Inserts the specified shape at position pos.

        If shape is a matrix, this matrix is inserted.
        If shape is a string, a shape is created according to the specified
        shape and color.
        """
        lowx, highx, lowy, highy = self.pos_to_coords(pos)
        if type(shape) == str:
            shape = self.make_shape(shape, color)
        if not self.is_empty(pos):
            if self.contains_only_agent(pos):
                self.delete_agent(pos, self.agent_state)
                self.grid[lowx:highx, lowy:highy, :] = shape
                self.shape_pos_list.append(pos)
                self.insert_agent(pos, self.agent_state)
            else:
                raise Warning('trying to insert a shape into a non-empty space, ' \
                    + 'command ignored')
                return
        else:
            self.grid[lowx:highx, lowy:highy, :] = shape
            self.shape_pos_list.append(pos)

    def delete_shape(self, pos):
        """
        Deletes all shape, agent included.
        """
        lowx, highx, lowy, highy = self.pos_to_coords(pos)
        self.grid[lowx:highx, lowy:highy, :] \
            = np.zeros((self.cellsize, self.cellsize, 3))
        self.shape_pos_list.remove(pos)

    def retrieve_shape(self, pos):
        lowx, highx, lowy, highy = self.pos_to_coords(pos)
        return np.array(self.grid[lowx:highx, lowy:highy, :])

    def move_shape(self, pos1, pos2):
        """
        Moves a shape from pos1 to pos2.
        If there is no shape at pos1, does nothing (raises a Warning).
        If there is already a shape at pos 2, does nothing (raises a Warning).
        """
        if self.is_empty(pos1):
            raise Warning('no shape to move at pos %s' % pos1)
            return
        if not self.is_empty(pos2):
            raise Warning('space at pos %s is not empty' % pos2)
            return
        shape = self.retrieve_shape(pos1)
        self.delete_shape(pos1)
        self.insert_shape(shape, pos2)

    def insert_random_shapes(self, n_shapes):
        intposlist = [self.gridsize*pos[0] + pos[1] for pos in \
            self.shape_pos_list]
        candidate_positions = [n for n in range(self.gridsize**2) if n not in \
            intposlist]
        new_positions = np.random.choice(candidate_positions, n_shapes, False)
        poss = []
        for intpos in new_positions:
            n = int(intpos // self.gridsize)
            m = int(intpos % self.gridsize)
            poss.append((n, m))
        color_array = [self.RED, self.GREEN, self.BLUE]
        cols = np.random.choice(3, n_shapes)
        shape_array = ['circle', 'square', 'diamond', 'triangle']
        shps = np.random.choice(4, n_shapes)
        for pos, col, shp in zip(poss, cols, shps):
            self.insert_shape(shape_array[shp], pos, color_array[col])

    def show_grid(self):
        plt.imshow(self.grid.swapaxes(0, 1))
        plt.show()

class GridLU(GridLUImage):
    """
    Class wrapper implementing the higher level GridLU functionnalities.
    Interface for the agent actions.
    """

    def __init__(self):
        super(GridLU, self).__init__()
        
        # self.insert_agent(self.agent_pos, self.agent_state)

        self.LEFT = np.array([-1, 0])
        self.RIGHT = np.array([1, 0])
        self.UP = np.array([0, -1])
        self.DOWN = np.array([0, 1])

        self.action_space = ['move_left',
                             'move_right',
                             'move_up',
                             'move_down',
                             'interact',
                             'no_op']
        self.num_actions = len(self.action_space)

    def is_valid_direction(self, direction):
        pos = self.agent_pos + direction
        n, m = int(pos[0]), int(pos[1])
        return n < self.gridsize and m < self.gridsize \
            and n >= 0 and m >= 0

    def move(self, direction):
        if not self.is_valid_direction(direction):
            return
        if self.contains_shape(self.agent_pos) and self.agent_state == 'active':
            # active and carrying a shape
            if self.is_empty(self.agent_pos + direction):
                # move only if there is no shape barring the way
                self.delete_agent(self.agent_pos, self.agent_state)
                shape = self.retrieve_shape(self.agent_pos)
                self.delete_shape(self.agent_pos)
                self.agent_pos += direction
                self.insert_shape(shape, self.agent_pos)
                self.insert_agent(self.agent_pos, self.agent_state)
            else:
                return  # cannot move with the attached shape because target
                      # location is not empty
        elif self.agent_state == 'passive':
            # if the agent is in a passive state, he can move wherever
            self.delete_agent(self.agent_pos, self.agent_state)
            self.agent_pos += direction
            self.insert_agent(self.agent_pos, self.agent_state)

    def move_left(self):
        self.move(self.LEFT)
    
    def move_right(self):
        self.move(self.RIGHT)

    def move_up(self):
        self.move(self.UP)

    def move_down(self):
        self.move(self.DOWN)

    def interact(self):
        if self.agent_state == 'passive' and self.contains_shape(self.agent_pos):
            # the agent can only become active to pick up an object
            self.delete_agent(self.agent_pos, self.agent_state)
            self.agent_state = 'active'
            self.insert_agent(self.agent_pos, self.agent_state)
        elif self.agent_state == 'active':
            self.delete_agent(self.agent_pos, self.agent_state)
            self.agent_state = 'passive'
            self.insert_agent(self.agent_pos, self.agent_state)

    def no_op(self):
        pass

    def sample_action(self):
        return np.random.choice(self.num_actions)

    def step(self, action):
        """
        Performs the specified action and returns the subsequent state.
        """
        if type(action) == str: 
            instruction = 'self.' + action + '()'
        if type(action) == int:
            instruction = 'self.' + self.action_space[action] + '()'
        eval(instruction)
        return self.grid

    def get_state(self):
        return self.grid

    def export_state_to_file(self):
        return NotImplemented

    def import_state_from_file(self):
        return NotImplemented

    def import_actions_and_play(self):
        """
        Imports a text file listing the actions, plays them in the environment
        and returns a list of images of the action execution.
        """
        return NotImplemented

# ============================= Test ======================================

# g = GridLU()
# g.random_start()
# text = input('type action here: ')
# while text != 'q':
#     if text in ['0', '1', '2', '3', '4', '5']:
#         text = int(text)
#         g.step(text)
#         print(g.agent_state)
#         g.show_grid()
#     text = input('type action here: ')