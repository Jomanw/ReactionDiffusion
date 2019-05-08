"""
Python Script for simulating Anisotropic Diffusion in a Reaction-Diffusion (RD) System.


Inspiration for various ways to use this system:
https://itb.biologie.hu-berlin.de/~bordyugov/tut/TBM2010/pdf/vl1.pdf

TODO: Incorporate stencil method for better derivatives (https://en.wikipedia.org/wiki/Five-point_stencil)

Other Helpful Links:
https://nbviewer.jupyter.org/github/barbagroup/CFDPython/blob/master/lessons/09_Step_7.ipynb
http://apmonitor.com/che263/index.php/Main/PythonDynamicSim
https://www.sympy.org/en/index.html



"""


import numpy as np
import sys


class TuringSystem():
    def __init__(self, grid_size=150, step_size=.01, num_elements=2, dt = .001):
        self.grid_size = grid_size
        self.step_size = step_size
        self.num_elements = num_elements
        self.grid = self.initialize_grid()
        self.D_x = 1
        self.D_y = 1

        # Diffusion Coefficients for x and y directions, for the Laplacian (second derivative)
        self.D_xx = 1
        self.D_yy = 1

        self.dt = dt
        self.max_val = 1000000000000

    # def initialize_grid_sine(self):


    def average_value(self):
        """
        Gets the average concentration values inside the grid, across all chemical species
        """
        return np.average(self.grid)

    def normalize_grid(self):
        """
        Scales the grid by the average value of the grid

        """
        self.grid = self.grid / self.average_value()

    def clip_grid(self):
        self.grid = np.clip(self.grid, -self.max_val, self.max_val)


    def initialize_grid(self):
        """
        Creates a grid of random size
        """
        grid = np.random.rand(self.grid_size, self.grid_size, self.num_elements)
        return grid

    def step(self, h):
        """
        Updates the internal grid
        """
        pass

    def viewable_grid(self):
        """
        Returns the grid without the edges padded
        """
        # return self.grid[1:-1, 1:-1]
        return self.grid

    def first_derivative(self, h):
        """
        Computes the first derivative with respect to space across a given grid.

        Does so using the centered-difference method. Error decays as the square of the step size.
        """
        # new_grid = np.zeros_like(grid)
        grid =  np.pad(self.grid, (1, 1), 'constant')[:, :, 1:-1]

        grid[0, :] = grid[1, :]
        grid[-1, :] = grid[-2, :]
        grid[:, 0] = grid[:, 1]
        grid[:, -1] = grid[:, -2]


        above = grid[0:-2, 1:-1]
        below = grid[2:, 1:-1]
        left = grid[1:-1, 0:-2]
        right = grid[1:-1, 2:]
        center = grid[1:-1, 1:-1]
        dx = self.D_x * (right - left) / (2 * h)
        dy = self.D_y * (above - below) / (2 * h)
        return dx + dy

    def second_derivative(self, h):
        """
        Computes the second derivative with respect to space across a given grid.

        Does so using the centered-difference method. Error decays as the square of the step size.

        Implements neumann boundary conditions as well.
        """
        grid =  np.pad(self.grid, (1, 1), 'constant')[:, :, 1:-1]

        grid[0, :] = grid[1, :]
        grid[-1, :] = grid[-2, :]
        grid[:, 0] = grid[:, 1]
        grid[:, -1] = grid[:, -2]


        above = grid[0:-2, 1:-1]
        below = grid[2:, 1:-1]
        left = grid[1:-1, 0:-2]
        right = grid[1:-1, 2:]
        center = grid[1:-1, 1:-1]
        dxx = self.D_xx * (left + right - 2 * center) / (h ** 2)
        dyy = self.D_yy * (above + below - 2 * center) / (h ** 2)
        return dxx + dyy

    def second_derivative_five_point_stencil(self, h):
        print(self.grid.shape)
        grid =  np.pad(self.grid, (2, 2), 'constant')[:, :, 2:-2]

        left = grid[2:-2, 1:-3]
        left_second = grid[2:-2, 0:-4]

        right = grid[2:-2, 3:-1]
        right_second = grid[2:-2, 4:]

        above = grid[1:-3, 2:-2]
        above_second = grid[0:-4, 2:-2]

        below = grid[3:-1, 2:-2]
        below_second = grid[4:, 2:-2]

        center = grid[2:-2, 2:-2]

        dxx = (-left_second + 16 * left - 30 * center + 16 * right - right_second) / (12 * h ** 2)
        dyy = (-below_second + 16 * below - 30 * center + 16 * above - above_second) / (12 * h ** 2)

        return dxx + dyy

class OriginalSystem(TuringSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, num_elements=2)

    def local_interactions_nonlinear(self):
        """
        Calculates nonlinear local interactions at each point within the grid.
        Comes from a Jupyter notebook found at: https://ipython-books.github.io/124-simulating-a-partial-differential-equation-reaction-diffusion-systems-and-turing-patterns/

        """
        grid = self.grid
        k = -.005
        tau = .1
        grid_copy = np.zeros_like(grid)
        grid_copy[:, :, 0] = grid[:, :, 0] - np.power(grid[:, :, 0], 3) - grid[:, :, 1] - k
        grid_copy[:, :, 1] = (grid[:, :, 0] - grid[:, :, 1])
        return grid_copy

    def step(self, h):
        """
        This is where the governing equations of the system are defined.
        """
        grid = self.grid
        # padded_grid = np.pad(grid, (1, 1), 'constant')[:, :, 1:-1]

        # Defines how the system will update over time

        du = self.dt * (self.second_derivative(h) + self.local_interactions_nonlinear()) * [1, 10]

        # Update the grid
        self.grid += du

class Schnakenberg(TuringSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.D_u = 1
        self.D_v = 10 
        self.gamma = 1000
        self.a = 0.126779
        self.b = 0.792366

    def local_interactions_nonlinear(self):
        u = self.grid[:, :, 0]
        v = self.grid[:, :, 1]
        grid_copy = np.zeros_like(self.grid)
        grid_copy[:, :, 0] = self.gamma * (self.a - u + (u ** 2) * v)
        grid_copy[:, :, 1] = self.gamma * (self.b - (u ** 2) * v)
        return grid_copy

    def step(self, h):
        du = self.dt * (self.second_derivative(h) * [self.D_u, self.D_v] + self.local_interactions_nonlinear())

        self.grid += du

class GM(TuringSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.D_u = .000945
        self.D_v = .27
        self.r = .001
        self.mu = 2.5
        self.alpha = 100

    def local_interactions_nonlinear(self):
        u = self.grid[:, :, 0]
        v = self.grid[:, :, 1]
        grid_copy = np.zeros_like(self.grid)
        grid_copy[:, :, 0] = self.r * u ** 2 / v - self.mu * u + self.r
        # u = grid_copy[:, :, 0]
        # print(u)
        grid_copy[:, :, 1] = self.r * u ** 2 - self.alpha * v
        return grid_copy

    def step(self, h):
        # padded_grid = np.pad(grid, (2, 2), 'constant')[:, :, 2:-2]
        du = self.dt * (self.second_derivative(h) * [self.D_u, self.D_v] + self.local_interactions_nonlinear())

        self.grid += du

class GMSC(TuringSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, num_elements=4)
        # print(self.grid.shape)
        # print(self.num_elements)
        self.D_u_gm = .000945
        self.D_v_gm = .27
        self.r = .001
        self.mu = 2.5
        self.alpha = 100

        self.D_u_sc = 1
        self.D_v_sc = 10
        self.gamma = 1000
        self.a = 0.126779
        self.b = 0.792366

        self.D_pigment_1 = 1
        self.D_pigment_2 = 1
        self.pigment_1_decay = .01
        self.pigment_2_decay = .01

    def local_interactions_nonlinear(self):
        u_gm = self.grid[:, :, 0]
        v_gm = self.grid[:, :, 1]
        u_sc = self.grid[:, :, 2]
        v_sc = self.grid[:, :, 3]

        # pigment_1 = self.grid[:, :, 4]
        # pigment_2 = self.grid[:, :, 5]

        grid_copy = np.zeros_like(self.grid)

        # GM Portion
        grid_copy[:, :, 0] = (self.r * u_gm ** 2 / v_gm - self.mu * u_gm + self.r) * (u_sc)
        grid_copy[:, :, 1] = (self.r * u_gm ** 2 - self.alpha * v_gm) * (v_sc)

        # SC Portion
        grid_copy[:, :, 2] = self.gamma * (self.a - u_sc + (u_sc ** 2) * v_sc)
        grid_copy[:, :, 3] = self.gamma * (self.b - (u_sc ** 2) * v_sc)

        # grid_copy[:, :, 4] = v_gm - self.pigment_1_decay * pigment_1 ** 2
        # grid_copy[:, :, 5] = u_sc - self.pigment_2_decay * pigment_2 ** 2

        return grid_copy

    def step(self, h):
        # padded_grid = np.pad(grid, (2, 2), 'constant')[:, :, 2:-2]
        du = self.dt * (self.second_derivative(h) * [self.D_u_gm, self.D_v_gm, self.D_u_sc, self.D_v_sc] + self.local_interactions_nonlinear())

        self.grid += du

class DoubleSC(TuringSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, num_elements=4)

        self.D_u_sc_1 = .1
        self.D_v_sc_1 = 1
        self.gamma = 1000
        self.a = 0.126779
        self.b = 0.792366

        self.D_u_sc_2 = 1
        self.D_v_sc_2 = 10
        self.gamma = 1000
        self.a = 0.126779
        self.b = 0.792366

    def local_interactions_nonlinear(self):
        u_sc_1 = self.grid[:, :, 0]
        v_sc_1 = self.grid[:, :, 1]
        u_sc_2 = self.grid[:, :, 2]
        v_sc_2 = self.grid[:, :, 3]

        # pigment_1 = self.grid[:, :, 4]
        # pigment_2 = self.grid[:, :, 5]

        grid_copy = np.zeros_like(self.grid)

        # First SC Portion
        grid_copy[:, :, 0] = self.gamma * (self.a - u_sc_1 + (u_sc_1 ** 2) * v_sc_1) * (u_sc_2)
        grid_copy[:, :, 1] = self.gamma * (self.b - (u_sc_1 ** 2) * v_sc_1) * (v_sc_2)

        # Second SC Portion
        grid_copy[:, :, 2] = self.gamma * (self.a - u_sc_2 + (u_sc_2 ** 2) * v_sc_2)
        grid_copy[:, :, 3] = self.gamma * (self.b - (u_sc_2 ** 2) * v_sc_2)

        # grid_copy[:, :, 4] = v_gm - self.pigment_1_decay * pigment_1 ** 2
        # grid_copy[:, :, 5] = u_sc - self.pigment_2_decay * pigment_2 ** 2

        return grid_copy

    def step(self, h):
        # padded_grid = np.pad(grid, (2, 2), 'constant')[:, :, 2:-2]
        du = self.dt * (self.second_derivative(h) * [self.D_u_sc_1, self.D_v_sc_1, self.D_u_sc_2, self.D_v_sc_2] + self.local_interactions_nonlinear())

        self.grid += du

class AnisotropicSchnakenberg(TuringSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.D_u = 1
        self.D_v = 10
        self.gamma = 1000
        self.a = 0.126779
        self.b = 0.792366
        self.D_xx = 20
        self.D_yy = .3

    def local_interactions_nonlinear(self):
        u = self.grid[:, :, 0]
        v = self.grid[:, :, 1]
        grid_copy = np.zeros_like(self.grid)
        grid_copy[:, :, 0] = self.gamma * (self.a - u + (u ** 2) * v)
        grid_copy[:, :, 1] = self.gamma * (self.b - (u ** 2) * v)
        return grid_copy

    def step(self, h):
        du = self.dt * (self.second_derivative(h) * [self.D_u, self.D_v] + self.local_interactions_nonlinear())

        self.grid += du

class Latch(TuringSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, num_elements=3)
        self.D_u = 1
        self.D_v = 10
        self.D_a = 50
        self.gamma = 1000
        self.a = 0.126779
        self.b = 0.792366

        self.diffusion_coefficient_array = [self.D_u, self.D_v, self.D_a][:self.num_elements]

        # self.num_iterations = 0
        # self.turning_point = 50

    def remove_activator(self):
        self.grid[:, :, 2] = np.zeros_like(self.grid[:, :, 2])

    def unexpose_activator(self):
        self.grid[:, 0:30, 2] = np.zeros_like(self.grid[:, 0:30, 2])

    def expose_activator(self, val=1):
        self.grid[:, 1, 2] += val

    def local_interactions_nonlinear(self):
        u = self.grid[:, :, 0]
        v = self.grid[:, :, 1]
        activator = self.grid[:, :, 2]
        inhibitor = self.grid[:, :, 3]

        grid_copy = np.zeros_like(self.grid)
        grid_copy[:, :, 0] = self.gamma * (self.a - u + (u ** 2) * v)
        grid_copy[:, :, 1] = self.gamma * (self.b - (u ** 2) * v)
        return grid_copy

    def step(self, h):
        # padded_grid = np.pad(grid, (2, 2), 'constant')[:, :, 2:-2]
        du = self.dt * (self.second_derivative(h) * self.diffusion_coefficient_array + self.local_interactions_nonlinear())

        self.grid += du

class Oscillatory(TuringSystem):

    def __init__(self, **kwargs):
        """
        https://people.maths.ox.ac.uk/maini/PKM%20publications/225.pdf

        """
        super().__init__(**kwargs)
        self.D = .516
        self.delta = 4
        self.alpha = .899
        self.beta = -.91
        self.r2 = 2
        self.r3 = 3.5


    def local_interactions_nonlinear(self):
        u = self.grid[:, :, 0]
        v = self.grid[:, :, 1]
        grid_copy = np.zeros_like(self.grid)
        grid_copy[:, :, 0] = self.alpha * u + v - self.alpha * self.r3 * u * v ** 2 - self.r2 * u * v
        grid_copy[:, :, 1] = self.beta * v - self.alpha * u + self.alpha * self.r3 * u * v ** 2 + self.r2 * u * v
        return grid_copy

    def step(self, h):
        # padded_grid = np.pad(grid, (2, 2), 'constant')[:, :, 2:-2]
        du = self.dt * (self.second_derivative(h) * [self.D * self.delta, self.delta] + self.local_interactions_nonlinear())

        self.grid += du



class FiveElementCoupled(TuringSystem):
    def __init__(self, **kwargs):
        """
        http://hopf.chem.brandeis.edu/pubs/pub288%20rep.pdf

        """
        super().__init__(**kwargs, num_elements=5)

        # # First Parameters:
        # self.D_x = .17
        # self.D_z = .17
        # self.D_r = 6
        # self.D_u = .5
        # self.D_w = 12
        # self.f = 1.4
        # self.f_bar = 1.1
        # self.q = .01
        # self.q_bar = .01
        # self.epsilon = .23
        # self.epsilon_bar = .5
        # self.delta = 2 * self.epsilon
        # self.delta_bar = 2 * self.epsilon_bar

        # # Second Parameters:
        self.D_x = .1
        self.D_z = .1
        self.D_r = .1
        self.D_u = 3
        self.D_w = 100
        self.f = 1.1
        self.f_bar = 0.65
        self.q = .01
        self.q_bar = .01
        self.epsilon = .215
        self.epsilon_bar = .5
        self.delta = 2 * self.epsilon
        self.delta_bar = 2 * self.epsilon_bar

    def F(self, x, z):
        return (1 / self.epsilon) * (x - x ** 2 - self.f * z * ((x - self.q) / (x + self.q)))

    def G(self, x, z):
        return x - z

    def F_bar(self, x, z):
        return (1 / self.epsilon_bar) * (x - x ** 2 - self.f_bar * z * ((x - self.q_bar) / (x + self.q_bar)))

    def local_interactions_nonlinear(self):
        x = self.grid[:, :, 0]
        z = self.grid[:, :, 1]
        r = self.grid[:, :, 2]
        u = self.grid[:, :, 3]
        w = self.grid[:, :, 4]

        grid_copy = np.zeros_like(self.grid)
        grid_copy[:, :, 0] = self.F(x, z) - (1 / self.delta) * (x - r)
        grid_copy[:, :, 1] = self.G(x, z)
        grid_copy[:, :, 2] = (1 / self.delta) * (x - r) + (1 / self.delta_bar) * (u - r)
        grid_copy[:, :, 3] = self.F_bar(u, w) - (1 / self.delta_bar) * (u - r)
        grid_copy[:, :, 4] = self.G(u, w)

        return grid_copy

    def step(self, h):
        # padded_grid = np.pad(grid, (2, 2), 'constant')[:, :, 2:-2]
        du = self.dt * (self.second_derivative(h) * [self.D_x, self.D_z, self.D_r, self.D_u, self.D_w] + self.local_interactions_nonlinear())

        self.grid += du

class SimpleSystem(TuringSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.D_u = 1
        self.D_v = 1
        self.a = 3
        self.b = 2
        self.c = 2
        self.d = 1

    def local_interactions_nonlinear(self):
        u = self.grid[:, :, 0]
        v = self.grid[:, :, 1]
        grid_copy = np.zeros_like(self.grid)
        grid_copy[:, :, 0] = self.a * u - self.b * v
        grid_copy[:, :, 1] = self.c * u - self.d * v
        return grid_copy

    def step(self, h):
        # padded_grid = np.pad(grid, (2, 2), 'constant')[:, :, 2:-2]
        du = self.dt * (self.second_derivative(h) * [self.D_u, self.D_v] + self.local_interactions_nonlinear())

        self.grid += du
