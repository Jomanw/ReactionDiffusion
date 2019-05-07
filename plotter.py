"""
Class to plot the grid

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os



class Plotter():
    def __init__(self, system, step_size, normalize = True, update_frequency = 100, save_name = None, animation = True, make_rgb = None):
        self.normalize = normalize
        self.system = system
        self.save_name = save_name
        self.num_elements = system.num_elements
        self.step_size = step_size
        self.make_rgb = make_rgb
        self.update_frequency = update_frequency

        if make_rgb != None:
            # assert len(make_rgb) == 3
            self.fig, self.axes = plt.subplots(1, self.num_elements + 1, figsize=(8, 4))
        else:
            self.fig, self.axes = plt.subplots(1, self.num_elements, figsize=(8, 4))
        plt.ion()

        self.animation_folder = "./animations/"
        self.frame_count = 0


    def update_drawing(self, make_rgb = None):
        """
        Updates the plot with the new grid
        """
        grid = self.system.viewable_grid()

        for axis_number in range(self.num_elements):
            ax = self.axes.flat[axis_number]
            ax.cla()
            ax.imshow(grid[:, :, axis_number],
            interpolation='bilinear',
            # vmin=0, vmax=2,
            extent=[-1, 1, -1, 1])

        if self.make_rgb != None:
            red_channel = grid[:, :, self.make_rgb[0]]
            normalized_red_channel = (red_channel - np.min(red_channel))/np.ptp(red_channel)


            if len(self.make_rgb) > 1:
                green_channel = grid[:, :, self.make_rgb[1]]
                normalized_green_channel = (green_channel - np.min(green_channel))/np.ptp(green_channel)
            else:
                normalized_green_channel =  np.zeros_like(grid[:, :, 0])

            if len(self.make_rgb) > 2:
                blue_channel = grid[:, :, self.make_rgb[2]]
                normalized_blue_channel = (blue_channel - np.min(blue_channel))/np.ptp(blue_channel)
            else:
                normalized_blue_channel =  np.zeros_like(grid[:, :, 0])

            rgb_data = np.stack((normalized_red_channel, normalized_green_channel, normalized_blue_channel),axis=2)

            ax = self.axes.flat[self.num_elements]
            ax.cla()
            ax.imshow(rgb_data,
            interpolation='bilinear',
            # vmin=0, vmax=1,
            extent=[-1, 1, -1, 1])

        # plt.draw()
        plt.pause(.0001)

    def get_unique_filename(self, filetype):
        count = 0
        while True:
            potential_name = self.save_name + str(count) + filetype
            if potential_name in os.listdir(self.animation_folder):
                count += 1
            else:
                return os.path.join(self.animation_folder, potential_name)

    def run(self, num_iterations):
        for iteration in range(num_iterations):
            self.system.step(self.step_size)
            if iteration % self.update_frequency == 0:
                self.update_drawing()
                print("Max Value: ", np.max(self.system.grid))
                print("Min Value: ", np.min(self.system.grid))
            # if self.normalize:
                # self.system.normalize_grid()
                # self.system.clip_grid()
        if self.save_name != None:
            for i in range(self.system.num_elements):
                filename = self.save_name + str(i) + ".png"
                data = self.system.grid[:, :, i]
                plt.imsave(filename, data)

    def make_animation(self, num_iterations, fps=15, frames = 300, filetype='.mp4'):
        # self.func(num_iterations)
        self.frame_count = 0
        self.animation = FuncAnimation(self.fig, self.func, frames=frames)
        filename = self.get_unique_filename(filetype)
        self.animation.save(filename, writer='imagemagick', fps=fps)


    def func(self, frame):
        self.frame_count += 1
        for i in range(self.update_frequency):
            self.system.step(self.step_size)
        self.update_drawing()
        print("Frame %i Computed."%(self.frame_count,))
