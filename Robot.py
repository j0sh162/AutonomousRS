import math
import numpy as np

CELL_SIZE = 1
SENSOR_RANGE = 850 # arbitrary
DELTA_T = .8
L = 24 # distance bewteen wheels

class Robot:
    
    position = (-30,-30)
    angle = 0
    sensors = []

    def __init__(self, position, angle):
        self.position = position
        self.angle = angle
        self.radius = L/2
        self.sensors = [Sensor(list(position),0,SENSOR_RANGE,self), Sensor(list(position), math.pi,SENSOR_RANGE,self),
                        Sensor(list(position),1,SENSOR_RANGE,self), Sensor(list(position),-1,SENSOR_RANGE,self), 
                        Sensor(list(position),0.5,SENSOR_RANGE,self), Sensor(list(position),-0.5,SENSOR_RANGE,self), 
                        Sensor(list(position),-1.5,SENSOR_RANGE,self), Sensor(list(position),1.5,SENSOR_RANGE,self),
                        Sensor(list(position),2,SENSOR_RANGE,self), Sensor(list(position),-2,SENSOR_RANGE,self), 
                        Sensor(list(position),-2.5,SENSOR_RANGE,self), Sensor(list(position),2.5,SENSOR_RANGE,self)]

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

    def update(self,map):
        pose = self.forward_kinematics(self.position[0],self.position[1],self.angle,1.1,1)
        v = [pose[0]-float(self.position[0]),float(pose[1]- self.position[1])]
        self.collision_check(map,v,pose[2])
        
        for sensor in self.sensors:
            sensor.update((float(self.position[0]),float(self.position[1])), map)
        return pose

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
            if int(map[y][x]) == 1:
                self.intersection_point = [x,y]
                return (x, y)
        self.intersection_point =[-30,-30]
        return None
    
    def update(self, starting_point, map):
        self.starting_point = starting_point
        self.ending_point = self.get_endpoint(starting_point,self.robot.angle+self.direction, self.length)
        self.update_intersection_point(map)
        self.distance = self.calculate_distance(self.starting_point[0],self.starting_point[1],self.intersection_point[0],self.intersection_point[1])
        points = self.get_points_on_line(self.starting_point,self.robot.angle+self.direction, 51,1)
        self.text_draw_point = [points[0][50],points[1][50]]
        # print(self.text_draw_point)
