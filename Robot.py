import math
import numpy as np


class Robot:

    l = 12 # distance between wheels in metres
    
    position = (0,0) # X, Y
    angle = 0
    sensors = []

    def __init__(self, position, angle):
        self.position = position
        self.angle = angle
        self.radius = self.l/2
        self.sensors = [Sensor(list(position),0,30,self)]

    def forward_kinematics(self, x, y, angle, Vl, Vr):
        delta_t = 0.01
        if abs(Vr - Vl) <= 1e-6:
            new_x = x + Vl * math.cos(angle) * delta_t
            new_y = y + Vl * math.sin(angle) * delta_t
            new_angle = angle  # No rotation
            return np.array([new_x, new_y, new_angle])
        else:
            R = (0.5*self.l)*((Vl+Vr)/(Vr-Vl)) 

        w = (Vr-Vl)/(self.l)

        icc = [x - R*math.sin(angle), y +R*math.cos(angle)]

        rotation = np.array([[math.cos(w*delta_t), -math.sin(w*delta_t), 0],
                             [math.sin(w*delta_t),  math.cos(w*delta_t), 0],
                             [0                  ,  0                  , 1]])
        
        # print(icc)
        second_part = np.array([x-icc[0],y-icc[1],angle])

        # print(rotation.shape)
        # print(second_part.shape)

        third_part = np.array([icc[0],icc[1],w*delta_t])
        # print(np.dot(rotation,second_part) + third_part)

       

        return np.dot(rotation,second_part) + third_part
    
    def collision_check(self,map,v,angle):
        x,y = self.position
        x_lower = math.floor(x - self.radius)
        y_lower = math.floor(y - self.radius)
        x_upper = math.ceil(x + self.radius)
        y_upper = math.ceil(y + self.radius)

        i = x_lower
        colliding_x = 0
        colliding_y = 0
        min_distance = float('inf')
        for i in range(x_upper):
            j = y_lower
            for j in range(y_upper):
                if(map[j][i] == 1):
                    dx = x - i
                    dy = y - j
                    for m in [[0,0.5],[0,-0.5],[0.5,0],[-0.5,0]]:
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
                                normalized_v = vec / np.sqrt(np.sum(vec**2))
                                move_vec = normalized_v*mov_dist


        self.angle = float(angle)

        if(min_distance == float('inf')):
            self.position = [self.position[0]+v[0],self.position[1]+v[1]]
        else:

            left_x = colliding_x - 0.5
            right_x = colliding_x + 0.5
            top_y = colliding_y - 0.5
            bottom_y = colliding_y + 0.5
            
            left_dist = self.distance(x,y,left_x,colliding_y)
            right_dist = self.distance(x,y,right_x,colliding_y)
            top_dist = self.distance(x,y,colliding_x,top_y)
            bot_dist = self.distance(x,y,colliding_x,bottom_y)

            min_dist = min([left_dist,right_dist,top_dist,bot_dist])

            n = None
            if(min_dist == left_dist):
                n = [-1,0]
                
            elif(min_dist == right_dist):
                n = [1,0]
                
            elif(min_dist == top_dist):
                n = [0,-1]
                
            else:
                n = [0,1]
            
            n = np.array(n)
            v_perp = (np.dot(np.array(v).T,n)*n)
            v_par = v - v_perp
            self.position = [self.position[0]+move_vec[0],self.position[1]+move_vec[1]]
            self.position = [self.position[0]+v_par[0],self.position[1]+v_par[1]]
      

    def distance(self,x1,y1,x2,y2):
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx * dx + dy * dy)

    def update(self,map):
        
        # logic stuff 
        pose = self.forward_kinematics(self.position[0],self.position[1],self.angle,3,3)
        # print(pose)
        v = [pose[0]-float(self.position[0]),float(pose[1]- self.position[1])]
        self.collision_check(map,v,pose[2])
        self.sensors[0].update((float(self.position[0]),float(self.position[1])), map)
        # print(self.position)
        print(pose[2])
        # self.position = (float(pose[0]), float(pose[1]))
    
        
        return pose

class Sensor:

    direction = 0
    length = 0
    robot = None
    starting_point = [0,0]
    ending_point = [0,0]

    def __init__(self, starting_point, direction, length, robot):
        self.length = length
        self.direction = direction
        self.robot = robot
        self.starting_point = starting_point
    
        
    def get_endpoint (self, start_point, angle_radians, length):
        """
        Draws a line in 2D space with:
        - start_point: (x0, y0)
        - angle_radians: direction in radians (0 = right, π/2 = up)
        - length: distance to extend the line
        - num_points: number of points for smoothness
        """
        x0, y0 = start_point
        
        # Direction vector (unit length)
        dx = np.cos(angle_radians)
        dy = np.sin(angle_radians)
        
        # Endpoint = start + length * direction
        end_x = x0 + length * dx
        end_y = y0 + length * dy

        return [float(end_x), float(end_y)]
    
       
    def get_points_on_line(self, start_point, angle_radians, length):

        x0, y0 = start_point
        
        # Direction vector (unit length)
        dx = np.cos(angle_radians)
        dy = np.sin(angle_radians)
        
        # Endpoint = start + length * direction
        end_x = x0 + length * dx
        end_y = y0 + length * dy

        num_points = self.get_num_points_between((x0,y0),(end_x,end_y), 1)
        x_values = np.linspace(x0, end_x, num_points)
        y_values = np.linspace(y0, end_y, num_points)

        return [x_values, y_values]

    def get_num_points_between(self, point1, point2, step_size=0.1):
        """
        Computes the number of points between two points given a step size.
        
        Args:
            point1 (tuple): (x1, y1)
            point2 (tuple): (x2, y2)
            step_size (float): Distance between consecutive points.
        
        Returns:
            int: Number of points.
        """
        x1, y1 = point1
        x2, y2 = point2
        
        # Calculate Euclidean distance
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Compute number of points
        num_points = int(distance / step_size) + 1
        
        return num_points
    
    def get_overlap_distance(self, map):
        print("test")
        points = self.get_points_on_line(self.starting_point,self.robot.angle+self.direction, self.length)
        # map is a 2d array with ones and zeros
        #array looks like this [[x1,x2...],[y1,y2...]]
        for i in range(len(points[0])):
            x = int(points[0][i])
            y = int(points[1][i])
            if int(map[y][x]) == 1:
                return (x, y)
            
        return None

    
    def update(self, starting_point, map):
        self.starting_point = starting_point
        self.ending_point = self.get_endpoint(starting_point,self.robot.angle+self.direction, self.length)
        print(self.get_overlap_distance(map))

def main():
    r = Robot((0,0), 0)

if __name__ == '__main__':
    main()
