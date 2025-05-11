import numpy as np
from Robot import Robot
import random
from math import pi as PI
import copy
from RunSimulation import read_bmp_with_pillow
from State import State

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
        self.prev_odo = self.curr_odo = self.get_state()
        self.p = [None] * num_particles
        init_grid = self.best_grid = np.full(self.grid_size, 0.5)

        y_max, x_max = self.grid_size
        for i in range(num_particles):
            x = random.uniform(0.0, x_max)
            y = random.uniform(0.0, y_max)
            self.p[i] = Robot((x,y),PI/2,init_grid)
        pass

    def fast_slam(self):
        self.update()
        self.curr_odo = self.get_state()
        z_star, free_grid_star, occupy_grid_star = self.sense()

        free_grid_offset_star = absolute2relative(free_grid_star, self.curr_odo)
        occupy_grid_offset_star = absolute2relative(occupy_grid_star, self.curr_odo)
        w = np.zeros(self.num_particles)
        for i in range(self.num_particles):
            p = self.p
            prev_pose = p[i].get_state()
            x, y, theta = p[i].motion_model()
            p[i].set_states([x, y, theta])
     
    
            # Calculate particle's weights depending on robot's measurement. ie lowest difference in measurment is best particle 
            z, _, _ = p[i].sense()
            w[i] = self.measurement_model(z_star, z)

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
        print(self.get_state())
        print('Estimate')
        print(estimated_R.get_state())

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
        return estimated_R

if __name__ == '__main__':
    world_grid = State(read_bmp_with_pillow('map2.bmp')).map
    robot = Fast_Slam((15,20),PI/2,world_grid,100)

    for i in range(100):
        estimate = robot.fast_slam()
        print(estimate.position)
        print(robot.position)