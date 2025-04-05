import pygame 
import random
import numpy as np
import struct
from State import State 

# Constants
WIDTH, HEIGHT = 1, 1
ROWS, COLS = 300, 300
scale = 1
CELL_SIZE = 1

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
# maze = [
#     [1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 1, 2, 0, 1],
#     [1, 0, 1, 0, 1, 1, 0, 1],
#     [1, 0, 1, 0, 0, 0, 0, 1],
#     [1, 0, 1, 0, 1, 0, 0, 1],
#     [1, 0, 1, 0, 1, 0, 0, 1],
#     [1, 0, 1, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1]
# ]

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
                val = 0
                if pixel[0] == 255:
                    val = 0
                else:
                    val = 1

                row.append(val)
            pixels.append(row)
    return pixels

# Goal at 206 140
def draw_grid(screen, grid):
    for y in range(ROWS):
        for x in range(COLS):
            color = WHITE if grid[y][x] == 0 else BLACK
            pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            # pygame.draw.line(screen,color,(x,y),(x+1,y+1),C)

def draw_robot(screen, robot):
    pygame.draw.circle(screen,RED,robot.position,(robot.radius))
    pygame.draw.line(screen,BLACK,robot.sensors[0].starting_point,robot.sensors[0].ending_point,1)
    # pygame.draw.circle(screen,RED,(96,96),4)

def main():
    pygame.init()
    screen = pygame.display.set_mode((ROWS, COLS))
    pygame.display.set_caption("Maze Generator")
    clock = pygame.time.Clock()
    state = State(read_bmp_with_pillow('map.bmp'))
    print(np.shape(state.map))
    #test

    running = True
    while running:
        screen.fill(WHITE)
        draw_grid(screen, state.map)
        draw_robot(screen,state.robot)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        pygame.display.flip()
        state.update()
        clock.tick(30)
    
    pygame.quit()
    
if __name__ == "__main__":
    main()
    # print(read_bmp_with_pillow('map.bmp'))

