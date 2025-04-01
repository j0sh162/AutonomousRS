import pygame
import random
import math

# Constants
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 20, 20
CELL_SIZE = WIDTH // COLS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)


def draw_maze(screen, grid):
    return None


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Maze Generator")
    clock = pygame.time.Clock()


    running = True
    while running:
        screen.fill(WHITE)
        # draw_maze(screen)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()
    
if __name__ == "__main__":
    main()

