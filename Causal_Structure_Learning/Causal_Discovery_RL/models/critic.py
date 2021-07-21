import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, DropoutWrapper
import numpy as np
#from tqdm import tqdm


class Critic(object):
 
 
    def __init__(self, config, is_train):
        self.config=config

        # Data config
        self.batch_size = config.batch_size 
        self.num_nodes = config.num_nodes 
        self.input_dimension = config.input_dimension 

        # Network config
        self.input_embed = config.d_model
        self.num_neurons = config.d_model
        self.initializer = tf.contrib.layers.xavier_initializer() 

        # Baseline setup
        self.init_baseline = 0.
 
    def predict_rewards(self, encoder_output):
        # [Batch size, Sequence Length, Num_neurons] to [Batch size, Num_neurons]
        frame = tf.reduce_mean(encoder_output, 1) 
 
        with tf.variable_scope("ffn"):
            # ffn 1
            h0 = tf.layers.dense(frame, self.num_neurons, activation=tf.nn.relu, kernel_initializer=self.initializer)
            # ffn 2
            w1 =tf.get_variable("w1", [self.num_neurons, 1], initializer=self.initializer)
            b1 = tf.Variable(self.init_baseline, name="b1")
            self.predictions = tf.squeeze(tf.matmul(h0, w1)+b1)