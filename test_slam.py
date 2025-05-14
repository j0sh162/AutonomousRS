from State import State
from fast_slam_robot import read_bmp_with_pillow

if __name__ == '__main__':
    world = State(read_bmp_with_pillow('map2.bmp'),(15,20))
    # robot = Fast_Slam((15,20),PI/2,world_grid,100)
    robot = world.robot
    for i in range(100):
        estimate = robot.fast_slam()
        print(estimate.position)
        print(robot.position)