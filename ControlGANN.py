import tensorflow.keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
from deap import base, creator, tools, algorithms
import pickle
import time
from PIL import Image
# from State import State


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


env = State(read_bmp_with_pillow('/content/drive/MyDrive/map2.bmp'))

in_dimen = 15 #Total no. of observations made about the environment
out_dimen = 2



def model_build(in_dimen=in_dimen,out_dimen=out_dimen):
    model = Sequential()
    model.add(Dense(12, input_dim=in_dimen, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(out_dimen, activation='tanh'))
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

model = model_build()

@tf.function
def fast_predict(model1, x):
  return model1(x, training=False)


def evaluate(individual,award=0):

    env.reset()     #Initiate the game
     #call the model
    # print("test")
    #set the weight of the model from the previous run of GA.
    #For the first run, it will be the random weight generated
    #model_weights_as_matrix function is to reshape the weights to make it compatible with the NN model
    #This function (and only this function) is borrowed from the PyGad source code
    #Detail of this function can be found in the Github link

    model.set_weights(model_weights_as_matrix(model, individual))

    # done = False  #Initial state of final outcome of the game is set to False which means game is not over
    step = 0  #Initiate the counter for no. of steps

    #Below first we have the stopping condition for each gameplay by each individual (chromosome).
    #The condition on step is given so that the game is not trapped somewhere and goes on forever
    # time_before = time.time()-start_time
    # print(time_before)
    while step<=30:

      #All the below steps are to  reshape the observation to make it the input layer of the NN
        #TODO implement full run to get reward.


        # In your loop
        state_tensor = tf.convert_to_tensor(np.asarray(env.getstate()).reshape(1, -1), dtype=tf.float32)
        output = fast_predict(model,state_tensor)[0].numpy() * 2

       

        for i in range(0,30):
          env.update(list(output))

        # np.asarray(env.getstate()).reshape(1, -1)
        step = step+1   #Increase the counter

    award = env.reward
    # print("eval time taken", time.time()-time_before-start_time)
    return (award,)


if __name__ == "__main__":
    # import multiprocessing
    start_time = time.time()
    # print("checkpoint 1", time.time()-start_time)
    model = model_build()
    ind_size = model.count_params()
    print(ind_size)
    print(model.summary())

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("weight_bin", np.random.uniform,-1,1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.weight_bin, n=ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("evaluate", evaluate)

    # pool = multiprocessing.Pool()
    # toolbox.register("map", pool.map)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Mean", np.mean)
    stats.register("Max", np.max)
    stats.register("Min", np.min)

    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)

    # print(np.asarray(env.getstate()).shape)
    # print(np.asarray(env.getstate()))
    # print(list(model.predict(np.asarray(env.getstate()).reshape(1, -1))[0]))
    # print("checkpoint 2", time.time()-start_time)

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox,
                                        mu=100,  # parents
                                        lambda_=100,  # offspring
                                        cxpb=0.8, mutpb=0.2,
                                        ngen=20,
                                        stats=stats,
                                        halloffame=hof,
                                        verbose=True)

    # print("checkpoint 3", time.time()-start_time)
    # pool.close()
    # pool.join()

    with open("robotmodel.pkl", "wb") as cp_file:
      pickle.dump(hof.items[0], cp_file)
