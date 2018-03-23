import os
import _pickle as cPickle
import numpy as np
import xml.etree.ElementTree as ET
import random
import time

import matplotlib.pyplot as plt
from utils import *
from model import *

# Model parameters
# Model parameters
class Param():
    def __init__(self):
        # General parameters
        self.train = 1 # Train the model
        self.sample = 0 # Sample from the model
        self.rnn_size = 256 # Size of RNN hidden state
        self.tsteps = 150 # RNN time steps (for backprop)
        self.nmixtures = 20 # Number of gaussian mixtures

        # Training parameters
        self.batch_size = 64 # Batch size for each gradient step
        self.nbatches = 500 # Number of batches per epoch, default is 500
        self.nepochs = 100 # Number of epochs, default is 250
        self.dropout = 0.95 # Probability of keeping neuron during dropout
        self.grad_clip = 10. # Clip gradients to this magnitude, default 10
        self.optimizer = 'rmsprop' # Ctype of optimizer: 'rmsprop' or 'adam'
        self.learning_rate = 1e-4 # Learning rate
        self.lr_decay = 1. # Decay rate for learning rate
        self.decay = 0.95 # Decay rate for rmsprop
        self.momentum = 0.9 # Momentum for rmsprop

        # Window parmaters
        self.kmixtures = 1 # Number of gaussian mixtures for character window
        self.alphabet = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' # Default is a-z, A-Z, space, and <UNK> tag
        self.tsteps_per_ascii = 25 # Expected number of pen points per character

        # Book-saving
        self.data_scale = 50 # Amount to scale data down before training
        self.log_dir ='./logs/' # Location, relative to execution, of log files
        self.data_dir ='./data' # Location, relative to execution, of data
        self.save_path ='saved/model.ckpt' # Location to save model
        self.save_every = 2000 # Number of batches between each save

        # Sampling
        self.text ='' # String for sampling model (defaults to test cases)
        self.style =-1 # Optionally condition model on a preset style (using data in styles.p)
        self.bias = 1.0 # Higher bias means neater, lower means more diverse (range is 0-5)
        self.sleep_time=60*5 # Time to sleep between running sampler
        
args = Param()

# Training the model

# Training the model

logger = Logger(args) # make logging utility
logger.write("\nTRAINING MODE...")
logger.write("{}\n".format(args))
logger.write("loading data...")
data_loader = DataLoader(args, logger=logger)

logger.write("building model...")
model = Model(args, logger=logger)

logger.write("attempt to load saved model...")
load_was_success, global_step = model.try_load_model(args.save_path)

v_x, v_y, v_s, v_c = data_loader.validation_data()
valid_inputs = {model.input_data: v_x, model.target_data: v_y, model.char_seq: v_c}

logger.write("training...")
model.sess.run(tf.assign(model.decay, args.decay ))
model.sess.run(tf.assign(model.momentum, args.momentum ))
running_average = 0.0 ; remember_rate = 0.99
training_time = time.time() # Used to compute global training time of the model

for e in range(int(global_step/args.nbatches), args.nepochs):
    model.sess.run(tf.assign(model.learning_rate, args.learning_rate * (args.lr_decay ** e)))
    logger.write("learning rate: {}".format(model.learning_rate.eval()))

    c0, c1, c2 = model.istate_cell0.c.eval(), model.istate_cell1.c.eval(), model.istate_cell2.c.eval()
    h0, h1, h2 = model.istate_cell0.h.eval(), model.istate_cell1.h.eval(), model.istate_cell2.h.eval()
    kappa = np.zeros((args.batch_size, args.kmixtures, 1))

    for b in range(global_step%args.nbatches, args.nbatches):

        i = e * args.nbatches + b
        if global_step is not 0 : i+=1 ; global_step = 0

        if i % args.save_every == 0 and (i > 0):
            model.time = model.time + time.time() - training_time
            model.saver.save(model.sess, args.save_path, global_step = i) ; 
            logger.write('SAVED MODEL.')
            training_time = time.time()
            print("The total time of training is:",model.time,"s.")
            
        start = time.time()
        x, y, s, c = data_loader.next_batch()

        feed = {model.input_data: x, model.target_data: y, model.char_seq: c, model.init_kappa: kappa, \
            model.istate_cell0.c: c0, model.istate_cell1.c: c1, model.istate_cell2.c: c2, \
            model.istate_cell0.h: h0, model.istate_cell1.h: h1, model.istate_cell2.h: h2}

        [train_loss, _] = model.sess.run([model.cost, model.train_op], feed)
        feed.update(valid_inputs)
        feed[model.init_kappa] = np.zeros((args.batch_size, args.kmixtures, 1))
        [valid_loss] = model.sess.run([model.cost], feed)

        running_average = running_average*remember_rate + train_loss*(1-remember_rate)

        end = time.time()
        if i % 10 is 0: logger.write("{}/{}, loss = {:.3f}, regloss = {:.5f}, valid_loss = {:.3f}, time = {:.3f}" \
            .format(i, args.nepochs * args.nbatches, train_loss, running_average, valid_loss, end - start))     