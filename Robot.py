import math
import numpy as np


class Robot:

    l = 8 # distance between wheels in metres
    
    position = (0,0) # X, Y

    def __init__(self, position, angle):
        self.position = position
        self.angle = angle
        self.radius = self.l/2

    def forward_kinematics(self, x, y, angle, Vl, Vr):
        delta_t = 1
        if abs(Vr - Vl) <= 1e-6:
            new_x = x + Vl * math.cos(angle) * delta_t
            new_y = y + Vl * math.sin(angle) * delta_t
            new_angle = angle  # No rotation
            return np.array([[new_x], [new_y], [new_angle]])
        else:
            R = (0.5*self.l)*((Vl+Vr)/(Vr-Vl)) 

        w = (Vr-Vl)/(self.l)

        icc = [x - R*math.sin(angle), y +R*math.cos(angle)]

       

        rotation = np.array([[math.cos(w*delta_t), -math.sin(w*delta_t), 0],
                             [math.sin(w*delta_t),  math.cos(w*delta_t), 0],
                             [0                  ,  0                  , 1]])
        
        # print(icc)
        second_part = np.array([[x-icc[0]],[y-icc[1]],[angle]])

        # print(rotation.shape)
        # print(second_part.shape)

        third_part = np.array([[icc[0]],[icc[1]],[w*delta_t]])
        # print(np.dot(rotation,second_part) + third_part)

       

        return np.dot(rotation,second_part) + third_part
    
    def collision_check(self,map,v):
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
                    dx = x - i;
                    dy = y - j;
                    distance = math.sqrt(dx * dx + dy * dy)
                    colliding = distance < self.radius
                    if(colliding):
                        if(distance <= min_distance):
                            min_distance = distance
                            colliding_x = i
                            colliding_y = j


            
        if(min_distance == float('inf')):
            pass
        else:
            left_x = colliding_x - 0.5
            right_x = colliding_x + 0.5
            top_y = colliding_y - 0.5
            bottom_y = colliding_y + 0.5
            
            left_dist = distance(x,y,left_x,colliding_y)
            right_dist = distance(x,y,right_x,colliding_y)
            top_dist = distance(x,y,colliding_x,top_y)
            bot_dist = distance(x,y,colliding_x,bottom_y)

            min_dist = min([left_dist,right_dist,top_dist,bot_dist])

            n
            if(min_dist == left_dist):
                n = [-1,0]
                
            elif(min_dist == right_dist):
                n = [1,0]
                
            elif(min_dist == top_dist):
                n = [0,-1]
                
            else:
                n = [0,1]
                
            v_perp = np.dot(np.dot(v,n),n)
            v_par = v - v_perp
      

    def distance(x1,y1,x2,y2):
        dx = x1 - x2;
        dy = y1 - y2;
        return math.sqrt(dx * dx + dy * dy)

    def update(self,map):
        
        # logic stuff 
        pose = self.forward_kinematics(self.position[0],self.position[1],self.angle,1,2)
        # print(pose)
        self.angle = float(pose[2])
        self.position = (float(pose[0]), float(pose[1]))
    
        
        return pose


def main():
    r = Robot((0,0), 0)
    # r.forward_kinematics(0,0,1,0,1)
    for i in range(0,50):
        print(r.update())

if __name__ == '__main__':
    main()





