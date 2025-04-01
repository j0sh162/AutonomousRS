from Robot import Robot


WIDTH, HEIGHT = (640*2,640*2)
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
        self.robot = Robot((96,96),0)

        # BOUNDS
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT    
        pass
    
    #TODO make efficient
    def collision(self,x,y):
        for i in range(len(maze)):
            for j in range(len(maze[0])):
                if(maze[i][j] == 1):
                    x_bound = i * CELL
                    y_bound = j * CELL
                    if(x_bound <= x and x <= x_bound + CELL and y_bound <= y and y <= y_bound + CELL):
                        return True

        return False
    
    def update(self):
        self.robot.update()
        print(state.robot.position)
        print(self.collision(self.robot.position[0],self.robot.position[1]))
        pass







if __name__ == "__main__":
    state = State()
    i  = 0
    while (i <= 100):
        state.update()
        i += 1

    
    print('Heloo')