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
        self.robot = Robot((96,96),PI/2)  

    
    #TODO make efficient


    
    def update(self):
        self.robot.update(self.map)
        pass

if __name__ == "__main__":
    state = State()
    i  = 0
    while (i <= 100):
        state.update()
        i += 1
