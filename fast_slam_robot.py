import numpy as np
from Robot import Robot
import random
from math import pi as PI
import copy
from PIL import Image

def read_bmp_with_pillow(file_path):
    # Open the image file using Pillow
    with Image.open(file_path) as img:
        # Ensure image is in RGB mode
        img = img.convert("RGB")
        width, height = img.size
        pixels = []

        # Load pixel data
        for y in range(height):
            row = []
            for x in range(width):
                pixel = img.getpixel((x, y))
                # If the red channel is 255 then consider that free space (0), else an obstacle (1)
                val = 0 if pixel[0] == 255 else 1
                row.append(val)
            pixels.append(row)
    return pixels

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


class Fast_Slam(Robot):
    def __init__(self,position,angle,grid,num_particles):
        super().__init__(position,angle,grid)
        self.num_particles = num_particles
        self.prev_odo = self.curr_odo = self.get_ordo()
        self.p = [None] * num_particles
        init_grid = self.best_grid = np.full(self.grid_size, 0.5)

        y_max, x_max = self.grid_size
        for i in range(num_particles):
            
            self.p[i] = Robot(self.position,PI/2,init_grid)
        self.estimation = self.p[0]
        
    def fast_slam(self,action):
        
        self.curr_odo = self.get_ordo()
        z_star, free_grid_star, occupy_grid_star = self.sense()

        free_grid_offset_star = absolute2relative(free_grid_star, self.curr_odo)
        occupy_grid_offset_star = absolute2relative(occupy_grid_star, self.curr_odo)
        w = np.zeros(self.num_particles)
        for i in range(self.num_particles):
            p = self.p
            prev_pose = p[i].get_ordo()
            x, y, theta = p[i].motion_model(action)
            p[i].set_states([x, y, theta])
     
    
            # Calculate particle's weights depending on robot's measurement. ie lowest difference in measurment is best particle 
            z, _, _ = p[i].sense()
            w[i] = self.measurement_model(z_star, z)

            # Update occupancy grid based on the true measurements
            
            curr_pose = p[i].get_ordo()
            free_grid = relative2absolute(free_grid_offset_star, curr_pose).astype(np.int32)
            occupy_grid = relative2absolute(occupy_grid_offset_star, curr_pose).astype(np.int32)
            p[i].update_occupancy_grid(free_grid, occupy_grid)

        # normalize
        w = w / np.sum(w)
        best_id = np.argsort(w)[-1]

        # select best particle
        self.estimation = copy.deepcopy(p[best_id])
   

        # Resample the particles with a sample probability proportional to the importance weight
        # Use low variance sampling method
        new_p = [None] * self.num_particles
        J_inv = 1 / self.num_particles
        r = random.random() * J_inv
        c = w[0]

        i = 0
        for j in range(self.num_particles):
            U = r + j * J_inv
            while (U > c):
                i += 1
                c += w[i]
            new_p[j] = copy.deepcopy(p[i])

        self.p = new_p
        self.prev_odo = self.curr_odo
        

    def update(self, action):
        out = super().update(action)
        self.fast_slam(action)
        return out