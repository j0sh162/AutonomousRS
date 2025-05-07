import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
from deap import base, creator, tools, algorithms
import pickle
from RobotEnviroment import RobotEnviroment


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

def evaluate(individual,award=0):
    env.reset()     #Initiate the game
    obs1 = env.state.robot.get_state() #Take the observation of the environment at the beginning of the game
    model = model_build()   #call the model

    #set the weight of the model from the previous run of GA. 
    #For the first run, it will be the random weight generated
    #model_weights_as_matrix function is to reshape the weights to make it compatible with the NN model
    #This function (and only this function) is borrowed from the PyGad source code
    #Detail of this function can be found in the Github link
    
    model.set_weights(model_weights_as_matrix(model, individual))   

    done = False  #Initial state of final outcome of the game is set to False which means game is not over
    step = 0  #Initiate the counter for no. of steps
    
    #Below first we have the stopping condition for each gameplay by each individual (chromosome). 
    #The condition on step is given so that the game is not trapped somewhere and goes on forever
    
    while (done == False) and (step<=1000): 
      
      #All the below steps are to  reshape the observation to make it the input layer of the NN
        #TODO implement full run to get reward.
        
        step = step+1   #Increase the counter
    return (award,)

model = model_build()
ind_size = model.count_params()
print(ind_size)
print(model.summary())

# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMax)
# toolbox = base.Toolbox()
# toolbox.register("weight_bin", np.random.uniform,-1,1)
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.weight_bin, n=ind_size)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
# toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("evaluate", evaluate)



# stats = tools.Statistics(lambda ind: ind.fitness.values)
# stats.register("Mean", np.mean)
# stats.register("Max", np.max)
# stats.register("Min", np.min)



# pop = toolbox.population(n=100)
# hof = tools.HallOfFame(1)


# pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.2, ngen=30, halloffame=hof, stats=stats)


# with open("lunarlander_model.pkl", "wb") as cp_file:

#     pickle.dump(hof.items[0], cp_file)
