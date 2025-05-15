import pygame 
import random
import numpy as np
import struct
from State import State 
from PIL import Image
from collections import deque
import tensorflow
import keras
from keras import Sequential
from keras._tf_keras.keras.layers import Dense
import pickle
# from ControlGANN import model_build, model_weights_as_matrix
import copy
actions = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 2.5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [2.5, 5], [2.5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [2.5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [2.5, 5], [2.5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
# Constants
WIDTH, HEIGHT = 1, 1
ROWS, COLS = 600, 600
scale = 1
CELL_SIZE = 1

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

pygame.init()
pygame.font.init() 
FONT = pygame.font.Font("ComicNeueSansID.ttf", 16)
# Makes visualisation nicer without affecting underlying logic
INTERPOLATION = False
# Shows the path of the robot 
TRAIL = True



# in_dimen = 15 #Total no. of observations made about the environment
# out_dimen = 2
def model_build(in_dimen=15,out_dimen=2):
    model = Sequential()
    model.add(Dense(12, input_dim=in_dimen, activation='relu'))   
    model.add(Dense(8, activation='relu'))
    model.add(Dense(out_dimen, activation='softmax'))
    #The compilation below is just declared. It is not used anywhere. That's why it does not matter which loss, optimizer or metrics we are using
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def model_weights_as_matrix(model, weights_vector):
    weights_matrix = []

    start = 0
    for layer_idx, layer in enumerate(model.layers): 
        layer_weights = layer.get_weights()
        if layer.trainable:
            for l_weights in layer_weights:
                layer_weights_shape = l_weights.shape
                layer_weights_size = l_weights.size
        
                layer_weights_vector = weights_vector[start:start + layer_weights_size]
                layer_weights_matrix = np.reshape(layer_weights_vector, newshape=(layer_weights_shape))
                weights_matrix.append(layer_weights_matrix)
        
                start = start + layer_weights_size
        else:
            for l_weights in layer_weights:
                weights_matrix.append(l_weights)

    return weights_matrix

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

stored_postion = deque(maxlen=5) # test this for diff values
senor_endpoints = []

def interpolation(deque):
    output = [0,0]
    x_values = [p[0] for p in deque]
    output[0] = int(sum(x_values) / len(x_values))
    y_values = [p[1] for p in deque]
    output[1] = int(sum(y_values) / len(y_values))
    return output

def draw_estimate(screen, robot, trail_counter, trail):
    if TRAIL and trail_counter == 0:
        trail.append((robot.estimation.position[0],robot.estimation.position[1]))
    for postion in trail:
        pygame.draw.circle(screen, (0,0,0), postion, 2)
    pygame.draw.circle(screen,GREEN,robot.estimation.position,robot.radius)


def draw_robot(screen, robot, trail_counter, trail):

    # Robot position interpolation 
    if INTERPOLATION:
        stored_postion.append([int(robot.position[0]),int(robot.position[1])])
        robot_draw_postion = interpolation(stored_postion)
    else:
        robot_draw_postion = robot.position

    # Drawing the trail behind the robot
    if TRAIL and trail_counter == 0:
        trail.append((robot.position[0],robot.position[1]))

    for postion in trail:
        pygame.draw.circle(screen, (0,0,0), postion, 2)


    if robot.sensors and len(robot.sensors) > 0:
        for i in range(1,len(robot.sensors)):
            # interpolation for sensor intersection points:
            if INTERPOLATION:
                senor_endpoints[i].append(robot.sensors[i].intersection_point)
                sensor_draw_position = interpolation(senor_endpoints[i])
            else:
                sensor_draw_position = robot.sensors[i].intersection_point
        
            pygame.draw.line(screen, BLACK,
                            robot_draw_postion,
                            sensor_draw_position, 1)
            pygame.draw.circle(screen, (0,130,100), sensor_draw_position,4)
            screen.blit(FONT.render(str(int(robot.sensors[i].distance)),True,(150,150,150)), robot.sensors[i].text_draw_point)
    pygame.draw.circle(screen, RED, robot_draw_postion, robot.radius)
    
    
    if INTERPOLATION:
        senor_endpoints[0].append(robot.sensors[0].intersection_point)
        sensor_draw_position = interpolation(senor_endpoints[0])
    else:
        sensor_draw_position = robot.sensors[0].intersection_point

    pygame.draw.line(screen, BLACK,
                            robot_draw_postion,
                            sensor_draw_position, 2)
    pygame.draw.circle(screen, (0,130,120), sensor_draw_position,4)

def draw_apples(state, screen):
    for apple in state.apple_locations:
        pygame.draw.circle(screen, (0,255,100), apple,4)

def create_grayscale_background(grid):
    """Create a grayscale version of the background.
    Grid values range from 0 (white) to 1 (black)."""
    background = pygame.Surface((COLS * CELL_SIZE, ROWS * CELL_SIZE))
    arr = copy.deepcopy(grid)

    mask_le_0_5 = arr <= 0.5

# Create a boolean mask where values are greater than 0.5
    mask_gt_0_5 = arr > 0.5

    # Assign 0 to elements where the value is less than or equal to 0.5
    arr[mask_le_0_5] = 0

    # Assign 1 to elements where the value is greater than 0.5
    arr[mask_gt_0_5] = 1
    for y in range(ROWS):
        for x in range(COLS):
            # Map the 0-1 value to a 255-0 grayscale value
            # 0 in grid = 255 grayscale (white)
            # 1 in grid = 0 grayscale (black)
            gray_value = int(255 * (1 - grid[y][x]))
            color = (gray_value, gray_value, gray_value)
            pygame.draw.rect(background, color, 
                            (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    return background

def main():

    # --- Step 2: Load the Best GA Individual (Flat Weights Vector) ---
    with open("robotmodel(3).pkl", "rb") as f:
        best_weights_vector = pickle.load(f)

    # --- Step 3: Rebuild Model Architecture ---
    model = model_build()

    # --- Step 4: Set the Best Weights into the Model ---
    model.set_weights(model_weights_as_matrix(model, best_weights_vector))

    trail_counter = 0
    trail = []
    # Use hardware acceleration and double buffering for better performance.
    trail_counter = 0
    trail = []
    estimate_trail = []
    
    # Create a single window that's twice as wide to hold both views
    combined_screen = pygame.display.set_mode(
        (COLS * CELL_SIZE * 2, ROWS * CELL_SIZE),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    pygame.display.set_caption("Simulation with Binary and Grayscale Views")
    
    # Create two surfaces for the different views
    binary_view = pygame.Surface((COLS * CELL_SIZE, ROWS * CELL_SIZE))
    grayscale_view = pygame.Surface((COLS * CELL_SIZE, ROWS * CELL_SIZE))
    
    clock = pygame.time.Clock()

    # Initialize state
    state = State(read_bmp_with_pillow('SimpleMap.bmp'), (100,100))
    if INTERPOLATION:
        for sensor in state.robot.sensors:
            senor_endpoints.append(deque(maxlen=5))

    print(np.shape(state.map))
    # 
    # Create both backgrounds
    binary_background = create_background_surface(state.map)

    running = True
    counter = 0
    inputs = np.asarray(state.getstate()).reshape(1, -1)
    # action_probs = model.predict(inputs, verbose=0)[0]


    while running:
        
        grayscale_background = create_grayscale_background(state.robot.estimation.grid)
        # Draw on both views
        binary_view.blit(binary_background, (0, 0))
        grayscale_view.blit(grayscale_background, (0, 0))
        
        draw_apples(state, binary_view)
        draw_robot(binary_view, state.robot, trail_counter, trail)
        
        draw_apples(state, grayscale_view)
        draw_estimate(grayscale_view, state.robot, trail_counter, estimate_trail)
        
        # Update TRAIL counter
        if TRAIL:
            trail_counter += 1
            if trail_counter == 1:
                trail_counter = 0
                
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Blit both views to the combined screen
        combined_screen.blit(binary_view, (0, 0))
        combined_screen.blit(grayscale_view, (COLS * CELL_SIZE, 0))
        
        pygame.display.flip()

        # Update logic [keep your existing code]
        if counter == 180:
            inputs = np.asarray(state.getstate()).reshape(1, -1)
            # action_probs = model.predict(inputs, verbose=0)[0]
            action_probs = [5,5]
            counter = 0
        print(counter)
        state.update(actions[counter])
        clock.tick(30)
        counter += 1
        

    pygame.quit()

if __name__ == "__main__":
    main()
