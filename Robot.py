import math
import numpy as np
import jax

CELL_SIZE = 1
SENSOR_RANGE = 850 # arbitrary this one is based on map size.
DELTA_T = .8
L = 24 # distance bewteen wheels

def is_within_radius(current_position, target_point, radius = 5):
  """
  Checks if a target point is within a given radius of the current position
  using squared distances for efficiency.

  Args:
    current_position (tuple): A tuple (x, y) representing the center point.
    target_point (tuple): A tuple (x, y) representing the point to check.
    radius (float or int): The radius. Must be non-negative.

  Returns:
    bool: True if the target point is within or on the boundary of the
          radius, False otherwise.

  Raises:
    ValueError: If the radius is negative.
  """
  if radius < 0:
      raise ValueError("Radius cannot be negative")

  x1, y1 = current_position
  x2, y2 = target_point

  # Calculate squared distance
  distance_sq = (x2 - x1)**2 + (y2 - y1)**2

  # Compare squared distance with squared radius
  return distance_sq <= radius**2


def logodds2prob(logodds):
    return 1 - 1 / (1 + np.exp(logodds) + 1e-15)


def prob2logodds(prob):
    return np.log(prob / (1 - prob + 1e-15))

def remove_point_from_list(list_of_points, point_to_remove):
    """
    Removes all occurrences of a specific point from a list of points.
    Returns a new list.
    """
    # Ensure point_to_remove is in a comparable format (e.g., tuple)
    # if your list_of_points contains tuples.
    # If list_of_points contains lists, point_to_remove should be a list.
    # For simplicity, let's assume consistent types or convert.
    if isinstance(point_to_remove, list):
        point_to_remove_tuple = tuple(point_to_remove)
    else:
        point_to_remove_tuple = point_to_remove # Assuming it's already a tuple

    new_list = []
    for p in list_of_points:
        # Convert current point p to tuple for consistent comparison
        if isinstance(p, list):
            current_point_tuple = tuple(p)
        else:
            current_point_tuple = p # Assuming it's already a tuple

        if current_point_tuple != point_to_remove_tuple:
            new_list.append(p) # Append the original point p
    return new_list

class Robot:
    beacons = [[87,431],[11,588],[588,437]]
    position = (-30,-30)
    angle = 0
    sensors = []


    def __init__(self, position, angle,grid):
        self.position = position
        self.angle = angle
        self.radius = L/2
        self.sensors = [Sensor(list(position),0,SENSOR_RANGE,self), Sensor(list(position), math.pi,SENSOR_RANGE,self),
                        Sensor(list(position),0.25,SENSOR_RANGE,self), Sensor(list(position), -0.25,SENSOR_RANGE,self),
                        Sensor(list(position),0.75,SENSOR_RANGE,self), Sensor(list(position), -0.75,SENSOR_RANGE,self),
                        Sensor(list(position),1.25,SENSOR_RANGE,self), Sensor(list(position), -1.25,SENSOR_RANGE,self),
                        Sensor(list(position),1.75,SENSOR_RANGE,self), Sensor(list(position), -1.75,SENSOR_RANGE,self),
                        Sensor(list(position),2.25,SENSOR_RANGE,self), Sensor(list(position), -2.25,SENSOR_RANGE,self),
                        Sensor(list(position),2.75,SENSOR_RANGE,self), Sensor(list(position), -2.75,SENSOR_RANGE,self),
                        Sensor(list(position),1,SENSOR_RANGE,self), Sensor(list(position),-1,SENSOR_RANGE,self), 
                        Sensor(list(position),0.5,SENSOR_RANGE,self), Sensor(list(position),-0.5,SENSOR_RANGE,self), 
                        Sensor(list(position),-1.5,SENSOR_RANGE,self), Sensor(list(position),1.5,SENSOR_RANGE,self),
                        Sensor(list(position),2,SENSOR_RANGE,self), Sensor(list(position),-2,SENSOR_RANGE,self), 
                        Sensor(list(position),-2.5,SENSOR_RANGE,self), Sensor(list(position),2.5,SENSOR_RANGE,self)]
        self.state_estimate = np.array((3))
        self.grid = grid
        self.grid_size = np.shape(self.grid)
        self.sensitivity_factor = 1.0
        self.occupy_prob = 0.8
        self.free_prob = 0.2
        self.prior_prob= 0.5


    def forward_kinematics(self, x, y, angle, Vl, Vr):
    
        if abs(Vr - Vl) <= 1e-6:
            new_x = x + Vl * math.cos(angle) * DELTA_T
            new_y = y + Vl * math.sin(angle) * DELTA_T
            new_angle = angle  # No rotation
            return np.array([new_x, new_y, new_angle])
        else:
            R = (0.5*L)*((Vl+Vr)/(Vr-Vl)) 

        w = (Vr-Vl)/(L)

        icc = [x - R*math.sin(angle), y +R*math.cos(angle)]

        rotation = np.array([[math.cos(w*DELTA_T), -math.sin(w*DELTA_T), 0],
                             [math.sin(w*DELTA_T),  math.cos(w*DELTA_T), 0],
                             [0                  ,  0                  , 1]])
        
        second_part = np.array([x-icc[0],y-icc[1],angle])

        third_part = np.array([icc[0],icc[1],w*DELTA_T])
        return np.dot(rotation,second_part) + third_part
    

    # TODO: Add noise to the respones to make it so that its more inline with real life 
    def motion_model(self,action):
        """
        Probabilistic motion model with noise.
        Returns a sampled next state based on forward kinematics plus noise.
        
        Args:
            action: [Vl, Vr] - Left and right wheel velocities
        
        Returns:
            np.array: Sampled next state [x, y, theta]
        """
        # Get nominal next state from forward kinematics
        nominal_next_state = self.forward_kinematics(
            self.position[0], self.position[1], self.angle, action[0], action[1]
        )
        
        # Motion noise parameters (these can be tuned)
        # Scale with magnitude of motion to make noise proportional to movement
        motion_magnitude = 0.5 * (abs(action[0]) + abs(action[1]))
        
        # Positional noise scales with movement
        alpha_xy = 0.05  # 10% positional noise
        sigma_xy = alpha_xy * motion_magnitude + 0.01  # Small base noise
        
        # Angular noise scales with rotation
        alpha_theta = 0.01 # 20% angular noise
        angular_change = abs(action[1] - action[0])  # Approximation of turning
        sigma_theta = alpha_theta * angular_change + 0.01  # Small base noise
        
        # Sample noise from Gaussian distributions
        noise_x = np.random.normal(0, sigma_xy)
        noise_y = np.random.normal(0, sigma_xy)
        noise_theta = np.random.normal(0, sigma_theta)
        
        # Add noise to nominal state
        noisy_state = nominal_next_state.copy()
        noisy_state[0] += noise_x
        noisy_state[1] += noise_y
        noisy_state[2] += noise_theta
        
        # Normalize angle to [-π, π]
        noisy_state[2] = np.mod(noisy_state[2] + np.pi, 2 * np.pi) - np.pi
        
        return noisy_state
    
    def get_ordo(self):
        return self.position[0],self.position[1],self.angle
    
        
    def set_states(self,pose):
        self.position = (pose[0],pose[1])
        self.angle = pose[2]

    
    def sense(self):
        occupied_points = []
        free_points = []
        distances = []
        for i in self.sensors:
            i.update(self.position, self.grid)
            
            x_int_candidates = np.round(i.intersection_point[0]).astype(int)
            y_int_candidates = np.round(i.intersection_point[1]).astype(int)
            point = np.array([x_int_candidates, y_int_candidates])
            occupied_points.append(point)
            
            # Get free points from this sensor and add them to our collection
            sensor_free_points = i.get_points_on_line_int()
            free_points.extend(sensor_free_points)
            
            distances.append(i.distance)
        
        # Convert to numpy arrays for easier indexing
        occupied_points = np.array(occupied_points)
        free_points = np.array(free_points)
        
        return distances, free_points, occupied_points

    
    # TODO: Add noise to the measurments for more realistic environment
    def measurement_model(self, z_star, z):
        """
        Calculates a weight based on the inverse of the sum of absolute differences.

        Args:
            z_star (np.array): Actual sensor measurements from the robot.
            z (np.array): Simulated sensor measurements from a particle.

        Returns:
            float: A weight for the particle. Higher for better matches.
        """
        z_star = np.array(z_star)
        z = np.array(z)

        if z_star.shape != z.shape:
            # Basic error handling or ensure they are always the same shape
            # For simplicity, if shapes mismatch, return a very low weight
            print(
                "Warning: z_star and z have different shapes in measurement model!"
            )
            return 1e-9

        # Calculate the sum of absolute differences for all sensor beams
        abs_diff = np.abs(z - z_star)
        sum_abs_diff = np.sum(abs_diff)

        # Calculate weight: inverse of (1 + scaled difference)
        # Adding 1 ensures that if diff is 0, weight is 1 (or max value).
        # The sensitivity_factor allows tuning how sharply the weight drops.
        weight = 1.0 / (1.0 + self.sensitivity_factor * sum_abs_diff)
        print(weight)
        return weight


    def collision_check(self,map,v,angle):
        x,y = self.position
        x_lower = math.floor(x - self.radius)
        y_lower = math.floor(y - self.radius)
        x_upper = math.ceil(x + self.radius)
        y_upper = math.ceil(y + self.radius)

        # i = x_lower
        colliding_x = 0
        colliding_y = 0
        min_distance = float('inf')
        for i in range(x_lower,x_upper):
            j = y_lower
            for j in range(y_lower,y_upper):
                if(map[j][i] == 1):
                    dx = x - i
                    dy = y - j
                    for m in [[0,0],[0,CELL_SIZE],[CELL_SIZE,0],[CELL_SIZE,CELL_SIZE]]:
                        dm_x = dx + m[0] 
                        dm_y = dy + m[1] 
                        distance = math.sqrt(dm_x * dm_x + dm_y * dm_y)
                        colliding = distance < self.radius
                        if(colliding):
                            if(distance <= min_distance):
                                min_distance = distance
                                colliding_x = i
                                colliding_y = j
                                mov_dist = self.radius - min_distance + 1/2
                                vec = np.array([dx,dy])
                                normalized_v = vec / np.sqrt(np.sum(np.square(vec)))
                                move_vec = normalized_v*mov_dist*0.7 # this is chosen to avoid too many visual bugs but may need to be disabled.


        self.angle = float(angle)

        if(min_distance == float('inf')):
            self.position = [self.position[0]+v[0],self.position[1]+v[1]]
        else:
            colliding_x += 0.5*CELL_SIZE
            colliding_y += 0.5*CELL_SIZE
            left_x = colliding_x - 0.5*CELL_SIZE
            right_x = colliding_x + 0.5*CELL_SIZE
            top_y = colliding_y - 0.5*CELL_SIZE
            bottom_y = colliding_y + 0.5*CELL_SIZE
            
            left_dist = self.distance(x,y,left_x,colliding_y)
            right_dist = self.distance(x,y,right_x,colliding_y)
            top_dist = self.distance(x,y,colliding_x,top_y)
            bot_dist = self.distance(x,y,colliding_x,bottom_y)

            min_dist = min([left_dist,right_dist,top_dist,bot_dist])

            n = None
            if(min_dist == left_dist):
                n = [-CELL_SIZE,0]
                
            elif(min_dist == right_dist):
                n = [CELL_SIZE,0]
                
            elif(min_dist == top_dist):
                n = [0,-CELL_SIZE]
                
            else:
                n = [0,CELL_SIZE]
            
            n = np.array(n)
            v_perp = (np.dot(np.array(v).T,n)*n)
            v_par = v - v_perp

            update = v_par+move_vec
            self.position = [self.position[0]+update[0],self.position[1]+update[1]]
            # self.position = [self.position[0]+v_par[0],self.position[1]+v_par[1]]

    def distance(self,x1,y1,x2,y2):
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx * dx + dy * dy)

    #TODO: Change the shit out of this 
    def update(self, action):
        pose = self.forward_kinematics(self.position[0],self.position[1],self.angle,action[0],action[1])
        
        v = [pose[0]-float(self.position[0]),float(pose[1]- self.position[1])]

        self.collision_check(self.grid,v,pose[2])
        
        self.collision_check(self.grid,v,pose[2])
        for sensor in self.sensors:
            sensor.update((float(self.position[0]),float(self.position[1])), self.grid)
        return pose

    def get_state(self):
        mystate = [self.position[0],self.position[1], self.angle]
        # print(len(self.sensors))
        for sensor in self.sensors:
            mystate.append(sensor.get_distance())

        return mystate
    
        
    def update_occupancy_grid(self, free_grid, occupy_grid):
        mask1 = np.logical_and(0 < free_grid[:, 0], free_grid[:, 0] < self.grid_size[1])
        mask2 = np.logical_and(0 < free_grid[:, 1], free_grid[:, 1] < self.grid_size[0])
        free_grid = free_grid[np.logical_and(mask1, mask2)]

        inverse_prob = self.inverse_sensing_model(False)
        l = prob2logodds(self.grid[free_grid[:, 1], free_grid[:, 0]]) + prob2logodds(inverse_prob) - prob2logodds(self.prior_prob)
        self.grid[free_grid[:, 1], free_grid[:, 0]] = logodds2prob(l)

        mask1 = np.logical_and(0 < occupy_grid[:, 0], occupy_grid[:, 0] < self.grid_size[1])
        mask2 = np.logical_and(0 < occupy_grid[:, 1], occupy_grid[:, 1] < self.grid_size[0])
        occupy_grid = occupy_grid[np.logical_and(mask1, mask2)]

        inverse_prob = self.inverse_sensing_model(True)
        l = prob2logodds(self.grid[occupy_grid[:, 1], occupy_grid[:, 0]]) + prob2logodds(inverse_prob) - prob2logodds(self.prior_prob)
        self.grid[occupy_grid[:, 1], occupy_grid[:, 0]] = logodds2prob(l)

    def inverse_sensing_model(self, occupy):
        if occupy:
            return self.occupy_prob
        else:
            return self.free_prob

class Sensor:

    distance = 1
    direction = 0
    length = 0
    robot = None
    starting_point = [0,0]
    ending_point = [0,0]
    intersection_point = [-30,-30]
    text_draw_point = [-30,-30]

    def __init__(self, starting_point, direction, length, robot):
        self.length = length
        self.direction = direction
        self.robot = robot
        self.starting_point = starting_point
    
    def calculate_distance(self,x1,y1,x2,y2):
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx * dx + dy * dy)

    def get_endpoint (self, start_point, angle_radians, length):
        x0, y0 = start_point

        dx = np.cos(angle_radians)
        dy = np.sin(angle_radians)
        # Endpoint = start + length * direction
        end_x = x0 + length * dx
        end_y = y0 + length * dy

        return [float(end_x), float(end_y)] 
    
    # TODO: make sure all the x_values and y_values are integer and non duplicate
    def get_points_on_line(self, start_point, angle_radians, length, resolution=1):
        x0, y0 = start_point
        
        dx = np.cos(angle_radians)
        dy = np.sin(angle_radians)
        
        end_x = x0 + length * dx
        end_y = y0 + length * dy

        num_points = self.get_num_points_between((x0,y0),(end_x,end_y), resolution)
        x_values = np.linspace(x0, end_x, num_points)
        y_values = np.linspace(y0, end_y, num_points)

        return [x_values, y_values]
        
    def get_points_on_line_int(self,resolution=1):
        x0, y0 = self.starting_point
        
        dx = np.cos(self.direction)
        dy = np.sin(self.direction)
        
        end_x = self.intersection_point[0]
        end_y = self.intersection_point[1]

        num_points = self.get_num_points_between((x0,y0),(end_x,end_y), resolution)
        x_values = np.linspace(x0, end_x, num_points)
        y_values = np.linspace(y0, end_y, num_points)

        x_int_candidates = np.round(x_values).astype(int)
        y_int_candidates = np.round(y_values).astype(int)

        # Store unique integer coordinate pairs, preserving order of appearance
        unique_ordered_points = []
        seen_coordinates = set()

        for x_val, y_val in zip(x_int_candidates, y_int_candidates):
            coord_tuple = (x_val, y_val)
            if coord_tuple not in seen_coordinates:
                unique_ordered_points.append(np.array([coord_tuple[0],coord_tuple[1]]))
                seen_coordinates.add(coord_tuple)


        return unique_ordered_points

    def get_num_points_between(self, point1, point2, step_size=1):
 
        x1, y1 = point1
        x2, y2 = point2
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        num_points = int(distance / step_size) + 1
        
        return num_points
    
    def update_intersection_point(self, map):
        points = self.get_points_on_line(self.starting_point,self.robot.angle+self.direction, self.length)
        for i in range(len(points[0])):
            x = int(points[0][i])
            y = int(points[1][i])
            map_height = len(map)
            map_width = len(map[0])
            x = max(0, min(x, map_width - 1))
            y = max(0, min(y, map_height - 1))
            if int(map[y][x]) == 1:
                self.intersection_point = [x,y]
                return (x, y)
        self.intersection_point =[-30,-30]
        return None
    
    def get_distance(self):
        return self.distance
    
    def update(self, starting_point, map):
        self.starting_point = starting_point
        self.ending_point = self.get_endpoint(starting_point,self.robot.angle+self.direction, self.length)
        self.update_intersection_point(map)
        self.distance = min(255.0, max(1, self.calculate_distance(self.starting_point[0],self.starting_point[1],self.intersection_point[0],self.intersection_point[1])))/255.0
        points = self.get_points_on_line(self.starting_point,self.robot.angle+self.direction, 51,1)
        self.text_draw_point = [points[0][50],points[1][50]]
        # print(self.text_draw_point)
