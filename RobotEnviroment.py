import random
import numpy as np
import struct
from State import State 
from PIL import Image
from collections import deque
import Robot
from math import pi
# Constants
# WIDTH, HEIGHT = 1, 1

# ROWS, COLS = 600, 600
# scale = 1
# CELL_SIZE = 1

class RobotEnviroment:

    state = None
    
    DEFAULT_START_LOCATION = (20,20) 

    def __init__(self, mapfilename):
        self.state = State(self.read_bmp_with_pillow(mapfilename), self.DEFAULT_START_LOCATION)

    def read_bmp_with_pillow(self, file_path):
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


    def step(self, action):
        self.state.robot.update(action)

    def reset(self):
        self.state.robot = Robot(self.DEFAULT_START_LOCATION,pi/2)
        