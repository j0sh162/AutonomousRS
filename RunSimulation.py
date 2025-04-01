import pygame 
import random
import numpy as np
from State import State 

# Constants
WIDTH, HEIGHT = 64, 64
ROWS, COLS = 8, 8
scale = 10
CELL_SIZE = (WIDTH // COLS)*scale

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
maze = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 2, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
]


def draw_grid(screen, grid):
    for y in range(ROWS):
        for x in range(COLS):
            color = WHITE if grid[y][x] == 0 else BLACK
            pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            # pygame.draw.line(screen,color,(x,y),(x+1,y+1),C)

def draw_robot(screen, robot):
    pygame.draw.circle(screen,RED,robot.position,(robot.radius)*100)

    # pygame.draw.circle(screen,RED,(96,96),4)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH*scale, HEIGHT*scale))
    pygame.display.set_caption("Maze Generator")
    clock = pygame.time.Clock()
    state = State()

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

