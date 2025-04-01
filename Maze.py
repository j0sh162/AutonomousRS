WIDTH, HEIGHT = 640*2
CELL = 64
MAZE = maze = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 2, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
]

class State():
    def __init__(self):
        self.map = MAZE
        self.robot = None
        # BOUNDS
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

        pass
    
    def bounds(self):
        pass
    def collision(self):
        
        pass
    







if __name__ == "__main__":
    print('Heloo')