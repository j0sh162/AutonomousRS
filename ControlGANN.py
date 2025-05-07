import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
from deap import base, creator, tools, algorithms
# import pickle
import RobotEnviroment

env = RobotEnviroment('map2.bmp')
env.reset()
in_dimen = 9 #Total no. of observations made about the environment
out_dimen = 2


def model_build(in_dimen=in_dimen,out_dimen=out_dimen):
    model = Sequential()
    model.add(Dense(12, input_dim=in_dimen, activation='relu'))   
    model.add(Dense(8, activation='relu'))
    model.add(Dense(out_dimen, activation='softmax'))
    #The compilation below is just declared. It is not used anywhere. That's why it does not matter which loss, optimizer or metrics we are using
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model
    



