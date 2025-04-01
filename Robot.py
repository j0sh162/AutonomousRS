import math
import numpy as np


class Robot:

    l = 0.16 # distance between wheels in metres
    
    position = (0,0) # X, Y
    angle = 0 

    def __init__(self, position, angle):
        self.position = position
        self.angle = angle
        

    def forward_kinematics(self, x, y, angle, Vl, Vr):
        
        R = (0.5*self.l)*((Vl+Vr)/(Vr-Vl)) 
        w = (Vr-Vl)/(self.l)

        icc = [x - R*math.sin(angle), y +R*math.cos(angle)]

        delta_t = 1

        rotation = np.array([[math.cos(w*delta_t), -math.sin(w*delta_t), 0],
                             [math.sin(w*delta_t),  math.cos(w*delta_t), 0],
                             [0                  ,  0                  , 1]])
        
        second_part = np.array([[x-icc[0]],[y-icc[1]],[angle]])

        third_part = np.array([[icc[0]],[icc[1]],[w*delta_t]])

        return np.dot(rotation,second_part) + third_part
    
    def update(self):
        # update location based on logic

        pose = self.forward_kinematics(self.position[0],self.position,1,0,1)    
        
        # return pose 
        return pose


def main():
    r = Robot()
    r.update()
    
if __name__ == '__main__':
    main()
