import math
import numpy as np


class Robot:

    l = 0.16 # distance between wheels in metres
    
    position = (0,0) # X, Y

    def __init__(self, position, angle):
        self.position = position
        self.angle = angle
        self.radius = self.l/2

    def forward_kinematics(self, x, y, angle, Vl, Vr):
        
        if Vr - Vl == 0:
            R = 1
        else:
            R = (0.5*self.l)*((Vl+Vr)/(Vr-Vl)) 

        w = (Vr-Vl)/(self.l)

        icc = [x - R*math.sin(angle), y +R*math.cos(angle)]

        delta_t = 1

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
    
    def update(self):
        
        # logic stuff 
        pose = self.forward_kinematics(self.position[0],self.position[1],self.angle,10,1)
        print(pose)
        self.angle = float(pose[2])
        self.position = (float(pose[0]), float(pose[1]))
    
        
        return pose


def main():
    r = Robot((0,0), 1)
    # r.forward_kinematics(0,0,1,0,1)
    for i in range(0,50):
        print(r.update())

if __name__ == '__main__':
    main()





