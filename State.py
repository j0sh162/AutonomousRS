from Robot import Robot
import random
import math
class State():


    apple_locations = []
    apple_reward= 0
    collision_r= 0
    exploration_r = 0
    robot_start_position = [25,25]


    def __init__(self,map):
        self.map = map
        self.robot = Robot(self.robot_start_position,3.141/2)
        # self.set_robot_start_position()
        self.apple_locations = self.generate_points(30)
        self.reward = 0
        self.visited_cells = {}       # Tracks visits to large regions
        self.cell_size = 30.0
        self.apple_reward = 0
        self.collision_r = 0
        self.exploration_r = 0

    def set_robot_start_position(self):
        position_valid = False

        rows = len(self.map)
        cols = len(self.map[0]) if rows > 0 else 0

        # Pick a random point (row, col)
        row = random.randint(0, rows - 1)
        col = random.randint(0, cols - 1)
        position = (col, row)
        x = position[0]
        y = position[1]

        while not position_valid:

            x_lower = math.floor(x - self.robot.radius)
            y_lower = math.floor(y - self.robot.radius)
            x_upper = math.ceil(x + self.robot.radius)
            y_upper = math.ceil(y + self.robot.radius)

            position_valid = True

            for i in range(x_lower, x_upper):
                for j in range(y_lower, y_upper):
                    if(self.map[j][i] == 1):
                        position_valid = False

            if(position_valid):
                self.robot_start_position = position
                self.robot = Robot(position,3.141/2)

    def calculate_distance(self,x1,y1,x2,y2):
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx * dx + dy * dy)

    def getstate(self):
        return self.robot.get_state()

    def generate_points(self, num_points):
        height = len(self.map)
        width = len(self.map[0])
        points = []

        while len(points) < num_points:
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            if self.map[y][x] == 0:  # 0 means walkable
                point = (x, y)
                if point not in points:  # Avoid duplicates
                    points.append(point)

        return points
    #TODO make efficient

    def update(self, action):
        self.robot.update(self.map, action)

        # Coarse discretization of position
        cell_x = int(self.robot.position[0] // self.cell_size)
        cell_y = int(self.robot.position[1] // self.cell_size)
        cell = (cell_x, cell_y)

        # Track visits to coarse cell
        if cell not in self.visited_cells:
            self.visited_cells[cell] = 0
        self.visited_cells[cell] += 1

        # Penalize repeated visits to the same large cell
        visit_count = self.visited_cells[cell]
        if visit_count > 5:
            self.exploration_r -= 0.2 * (visit_count - 5)

        x, y = self.robot.position
        x_lower = math.floor(x - self.robot.radius)
        y_lower = math.floor(y - self.robot.radius)
        x_upper = math.ceil(x + self.robot.radius)
        y_upper = math.ceil(y + self.robot.radius)

        # If apple_locations is a list, avoid modifying it while iterating
        to_remove = []

        for i in range(x_lower, x_upper):
            for j in range(y_lower, y_upper):
                if (i, j) in self.apple_locations:
                    self.apple_reward += 300
                    to_remove.append((i, j))

                if(self.map[j][i] == 1):
                    self.collision_r -= 2

        # d = self.calculate_distance(self.robot_start_position[0], self.robot_start_position[1], self.robot.position[0],self.robot.position[1])
        # self.reward += math.log(1 + d)

        for apple in to_remove:
            self.apple_locations.remove(apple)

        # print(self.reward)

    def reset(self):
        self.robot = Robot([25,25],3.141/2)
        self.apple_locations = self.generate_points(30)
        self.reward = 0
        self.visited_cells = {}
        self.apple_reward = 0
        self.collision_r = 0
        self.exploration_r = 0

    def get_state_fitness(self):
      return self.apple_reward +  self.collision_r + self.exploration_r
