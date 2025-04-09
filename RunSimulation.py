import pygame 
import random
import numpy as np
import struct
from State import State 
from PIL import Image

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
FONT = pygame.font.Font("ComicNeueSansID.ttf", 20)



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

def draw_robot(screen, robot):
    # Draw the robot's sensor line (if available)
    if robot.sensors and len(robot.sensors) > 0:
        for i in range(0,len(robot.sensors)):
            pygame.draw.line(screen, BLACK,
                            robot.sensors[i].starting_point,
                            robot.sensors[i].intersection_point, 1)
            pygame.draw.circle(screen, (0,255,00), robot.sensors[i].intersection_point,4)
            screen.blit(FONT.render(str(int(robot.sensors[i].distance)),True,(150,150,150)), robot.sensors[i].intersection_point)
    pygame.draw.circle(screen, RED, robot.position, robot.radius)#
    pygame.draw.line(screen, BLACK,
                            robot.sensors[0].starting_point,
                            robot.sensors[0].intersection_point, 1)

def main():
    
    # Use hardware acceleration and double buffering for better performance.
    screen = pygame.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE),
                                     pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Maze Generator")
    clock = pygame.time.Clock()

    # Initialize state and cache the static grid as background.
    state = State(read_bmp_with_pillow('map2.bmp'))
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
