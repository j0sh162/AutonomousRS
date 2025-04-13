import pygame 
import random
import numpy as np
import struct
from State import State 
from PIL import Image
from collections import deque

# Constants
WIDTH, HEIGHT = 1, 1
ROWS, COLS = 600, 600
scale = 1
CELL_SIZE = 1

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

pygame.init()
pygame.font.init() 
FONT = pygame.font.Font("ComicNeueSansID.ttf", 16)
INTERPOLATION = True


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

def create_background_surface(grid):
    """Pre-render the static grid to a surface."""
    background = pygame.Surface((COLS * CELL_SIZE, ROWS * CELL_SIZE))
    for y in range(ROWS):
        for x in range(COLS):
            color = WHITE if grid[y][x] == 0 else BLACK
            pygame.draw.rect(background, color, 
                             (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    return background

stored_postion = deque(maxlen=5) # test this for diff values
senor_endpoints = []

def interpolation(deque):
    output = [0,0]
    x_values = [p[0] for p in deque]
    output[0] = int(sum(x_values) / len(x_values))
    y_values = [p[1] for p in deque]
    output[1] = int(sum(y_values) / len(y_values))
    return output


def draw_robot(screen, robot):

    # Robot position interpolation 
    if INTERPOLATION:
        stored_postion.append([int(robot.position[0]),int(robot.position[1])])
        robot_draw_postion = interpolation(stored_postion)
    else:
        robot_draw_postion = robot.position

    if robot.sensors and len(robot.sensors) > 0:
        for i in range(1,len(robot.sensors)):
            # interpolation for sensor intersection points:
            if INTERPOLATION:
                senor_endpoints[i].append(robot.sensors[i].intersection_point)
                sensor_draw_position = interpolation(senor_endpoints[i])
            else:
                sensor_draw_position = robot.sensors[i].intersection_point
        
            pygame.draw.line(screen, BLACK,
                            robot_draw_postion,
                            sensor_draw_position, 1)
            pygame.draw.circle(screen, (0,255,100), sensor_draw_position,4)
            screen.blit(FONT.render(str(int(robot.sensors[i].distance)),True,(150,150,150)), robot.sensors[i].text_draw_point)
    pygame.draw.circle(screen, RED, robot_draw_postion, robot.radius)#
    
    if INTERPOLATION:
        senor_endpoints[0].append(robot.sensors[0].intersection_point)
        sensor_draw_position = interpolation(senor_endpoints[0])
    else:
        sensor_draw_position = robot.sensors[0].intersection_point

    pygame.draw.line(screen, BLACK,
                            robot_draw_postion,
                            sensor_draw_position, 2)
    pygame.draw.circle(screen, (0,255,100), sensor_draw_position,4)
def main():
    
    # Use hardware acceleration and double buffering for better performance.
    screen = pygame.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE),
                                     pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Maze Generator")
    clock = pygame.time.Clock()

    # Initialize state and cache the static grid as background.
    state = State(read_bmp_with_pillow('map2.bmp'))
    if INTERPOLATION:
        for sensor in state.robot.sensors:
            senor_endpoints.append(deque(maxlen=5))

    print(np.shape(state.map))
    background = create_background_surface(state.map)
    running = True
    while running:
        # Blit the cached background instead of drawing each cell per frame.
        screen.blit(background, (0, 0))
        draw_robot(screen, state.robot)

        # Process events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.flip()
        state.update()
        clock.tick(60)  # Cap the frame rate to 60 FPS

    pygame.quit()

if __name__ == "__main__":
    main()
