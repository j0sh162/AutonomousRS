from Robot import Robot
import random
import math
class State():

    #TODO put points all over the space and check if the robot has touched them if he touched them remove from list them and to reward
    apple_locations = []
    reward = 0
    robot_start_postion = [25,25]
    

    def __init__(self,map, robot_start_postion):
        self.map = map
        self.robot = Robot(robot_start_postion,3.141/2)  
        self.apple_locations = self.generate_points(30)
        self.reward = 0

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
                    self.reward += 1
                    to_remove.append((i, j))
                    # print("APPLE FOUND")
        for apple in to_remove:
            self.apple_locations.remove(apple)

        # print(self.reward)

    def reset(self):
        self.robot = Robot(self.robot_start_postion,3.141/2)  
        self.apple_locations = self.generate_points(30)
        self.reward = 0
    
if __name__ == "__main__":
    state = State()
    i  = 0
    while (i <= 100):
        state.update()
        i += 1
