from keras.models import load_model
from utils import plot_stroke
import random as rd
import numpy as np

seeds = np.load('../data/X_test.npy', allow_pickle=True)
unconditional_model = load_model('../models/unconditional_generation_model_double_output_best.h5')
p = 0.038509945441853315


def generate_unconditionally(random_seed=1):

    gen_len = 400 + rd.randint(0, 50)
    gen_stroke = []
    
    gen_buffer = seeds[random_seed]
    
    start_gen = 0
    i = 0
    
    while start_gen == 0 and i <1000:
        new_step = unconditional_model.predict(np.array([gen_buffer]))
        if rd.random() <= p :
            new_step[0][0][0] = 1
            start_gen = 1
        else :
            new_step[0][0][0] = 0
        new_step = np.array([new_step[0][0][0], new_step[1][0][0], new_step[1][0][1]])
        new_step = np.reshape(new_step, newshape=(1, 3))
        gen_buffer = gen_buffer[1:]
        gen_buffer = np.append(gen_buffer, new_step, axis=0)
        i+=1
    
    stop_gen = 0
    i=0
    while i <gen_len:
        new_step = unconditional_model.predict(np.array([gen_buffer]))
        if rd.random() <= p :
            new_step[0][0][0] = 1
            if i >= gen_len:
                stop_gen = 1
        else :
            new_step[0][0][0] = 0
        new_step = np.array([new_step[0][0][0], new_step[1][0][0], new_step[1][0][1]])
        new_step = np.reshape(new_step, newshape=(1, 3))
        gen_buffer = gen_buffer[1:]
        gen_buffer = np.append(gen_buffer, new_step, axis=0)
        gen_stroke.append(new_step[0])
        i+=1
    gen_stroke = np.array(gen_stroke)
    return gen_stroke


def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'