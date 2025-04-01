from Robot import Robot
PI = 3.14


CELL = 1
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
    def __init__(self,map):
        self.map = map
        self.robot = Robot((96,96),PI/4)  
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
        pass

if __name__ == "__main__":
    state = State()
    i  = 0
    while (i <= 100):
        state.update()
        i += 1
