import pygame
import random
import numpy as np

# Constants
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 20, 20
CELL_SIZE = WIDTH // COLS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Directions
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left

def create_grid():
    return [[1 for _ in range(COLS)] for _ in range(ROWS)]

def carve_maze(grid, x, y, visited):
    visited.add((x, y))
    grid[y][x] = 0
    random.shuffle(DIRECTIONS)
    
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx * 2, y + dy * 2
        if 0 <= nx < COLS and 0 <= ny < ROWS and (nx, ny) not in visited:
            grid[y + dy][x + dx] = 0
            carve_maze(grid, nx, ny, visited)

def draw_grid(screen, grid):
    for y in range(ROWS):
        for x in range(COLS):
            color = WHITE if grid[y][x] == 0 else BLACK
            pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Maze Generator")
    clock = pygame.time.Clock()
    
    grid = create_grid()
    carve_maze(grid, 0, 0, set())
    print(grid)
    print(np.shape(grid))
    running = True
    while running:
        screen.fill(WHITE)
        draw_grid(screen, grid)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        pygame.display.flip()

        clock.tick(30)
    
    pygame.quit()
    
if __name__ == "__main__":
    main()

