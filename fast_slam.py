import numpy as np
import random
import copy
import os
import argparse
import yaml

from State import State
from Robot import Robot
# from motion_model import MotionModel
# from measurement_model import MeasurementModel
from RunSimulation import read_bmp_with_pillow
import random

from math import pi as PI


def normalDistribution(mean, variance):
    return np.exp(-(np.power(mean, 2) / variance / 2.0) / np.sqrt(2.0 * np.pi * variance))

def create_rotation_matrix(theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    R_inv = np.linalg.inv(R)

    return R, R_inv


def absolute2relative(position, states):
    x, y, theta = states
    pose = np.array([x, y])

    R, R_inv = create_rotation_matrix(theta)
    position = position - pose
    position = np.array(position) @ R_inv.T

    return position


def relative2absolute(position, states):
    x, y, theta = states
    pose = np.array([x, y])

    R, R_inv = create_rotation_matrix(theta)
    position = np.array(position) @ R.T
    position = position + pose

    return position

if __name__ == "__main__":
    NUMBER_OF_PARTICLES = 100

    # create a world map

    world_grid = State(read_bmp_with_pillow('map2.bmp')).map

    # create a robot

    R = Robot((15,20),PI/2,world_grid) 
    prev_odo = curr_odo = R.get_state()

    # initialize particles
    p = [None] * NUMBER_OF_PARTICLES
 
    init_grid = np.full(R.grid_size, 0.5)
    
    y_max, x_max = R.grid_size
    for i in range(NUMBER_OF_PARTICLES):
        x = random.uniform(0.0, x_max)
        y = random.uniform(0.0, y_max)
        p[i] = Robot((x,y),PI/2,init_grid)

    # create motion model
    motion_model = R.motion_model

    # create measurement model
    measurement_model = R.measurement_model

    # FastSLAM1.0
    for count in range(100):

        R.update()
        curr_odo = R.get_state()
        z_star, free_grid_star, occupy_grid_star = R.sense()
        
        free_grid_offset_star = absolute2relative(free_grid_star, curr_odo)
        occupy_grid_offset_star = absolute2relative(occupy_grid_star, curr_odo)

        w = np.zeros(NUMBER_OF_PARTICLES)
        for i in range(NUMBER_OF_PARTICLES):

            prev_pose = p[i].get_state()
            x, y, theta = p[i].motion_model()
            p[i].set_states([x, y, theta])
     
    
            # Calculate particle's weights depending on robot's measurement. ie lowest difference in measurment is best particle 
            z, _, _ = p[i].sense()
            w[i] = measurement_model(z_star, z)

            # Update occupancy grid based on the true measurements
            
            curr_pose = p[i].get_state()
            free_grid = relative2absolute(free_grid_offset_star, curr_pose).astype(np.int32)
            occupy_grid = relative2absolute(occupy_grid_offset_star, curr_pose).astype(np.int32)
            p[i].update_occupancy_grid(free_grid, occupy_grid)

        # normalize
        w = w / np.sum(w)
        best_id = np.argsort(w)[-1]

        # select best particle
        estimated_R = copy.deepcopy(p[best_id])
        print('Real')
        print(R.get_state())
        print('Estimate')
        print(estimated_R.get_state())

        # Resample the particles with a sample probability proportional to the importance weight
        # Use low variance sampling method
        new_p = [None] * NUMBER_OF_PARTICLES
        J_inv = 1 / NUMBER_OF_PARTICLES
        r = random.random() * J_inv
        c = w[0]

        i = 0
        for j in range(NUMBER_OF_PARTICLES):
            U = r + j * J_inv
            while (U > c):
                i += 1
                c += w[i]
            new_p[j] = copy.deepcopy(p[i])

        p = new_p
        prev_odo = curr_odo

        #TODO: Integrate this with the UI 
        # Current IDEA could be to refactor this to super the orginal robot class then we just make a fast slam robot 
        # and for the particles we can just set it up with the original robot class

