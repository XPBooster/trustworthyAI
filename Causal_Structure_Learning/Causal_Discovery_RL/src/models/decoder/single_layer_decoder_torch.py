import tensorflow as tf
from tensorflow.contrib import distributions as distr
import torch
import torch.nn as nn
import torch.distributions.bernoulli
import torch.nn.functional as F
class SingleLayerDecoder(object):

    def __init__(self, config, is_train):
        self.batch_size = config.batch_size    # batch size
        self.max_length = config.max_length    # input sequence length (number of cities)
        self.input_embed = config.hidden_dim    # dimension of embedding space (actor)
        self.decoder_hidden_dim = config.decoder_hidden_dim
        self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer
        self.decoder_activation = config.decoder_activation
        self.use_bias = config.use_bias
        self.bias_initial_value = config.bias_initial_value
        self.use_bias_constant = config.use_bias_constant

        self.is_training = is_train

        self.samples = []
        self.mask = 0
        self.mask_scores = []
        self.entropy = []

    def decode(self, x):
        # encoder_output is a tensor of size [batch_size, max_length, input_embed]
        # with tf.variable_scope('singe_layer_nn'):
        W_l = torch.nn.Parameter(nn.init.xavier_normal_(torch.empty(self.input_embed, self.decoder_hidden_dim)))
        W_r = torch.nn.Parameter(nn.init.xavier_normal_(torch.empty(self.input_embed, self.decoder_hidden_dim)))
        W_u = torch.nn.Parameter(nn.init.xavier_normal_(torch.empty(self.decoder_hidden_dim)))    # Aggregate across decoder hidden dim


        dot_l = torch.einsum('ijk, kl->ijl', x, W_l)
        dot_r = torch.einsum('ijk, kl->ijl', x, W_r)

        tiled_l = torch.unsqueeze(dot_l, dim=2)
        tiled_l = tiled_l.expand(-1, -1, self.max_length, -1)
        tiled_r = torch.unsqueeze(dot_r, dim=2)
        tiled_r = tiled_r.expand(-1, -1, self.max_length, -1)
        if self.decoder_activation == 'tanh':    # Original implementation by paper
            final_sum = F.tanh(tiled_l + tiled_r)
        elif self.decoder_activation == 'relu':
            final_sum = F.relu(tiled_l + tiled_r)
        elif self.decoder_activation == 'none':    # Without activation function
            final_sum = tiled_l + tiled_r
        else:
            raise NotImplementedError('Current decoder activation is not implemented yet')

        # final_sum is of shape (batch_size, max_length, max_length, decoder_hidden_dim)
        logits = torch.einsum('ijkl, l->ijk', final_sum, W_u)    # batch, max_len, max_len



        if self.use_bias:    # Bias to control sparsity/density
            if self.bias_initial_value is None:  # Randomly initialize the learnable bias
                self.logit_bias = torch.nn.Parameter(nn.init.xavier_normal_(torch.empty(1)))
            elif self.use_bias_constant:  # Constant bias
                self.logit_bias = torch.tensor([self.bias_initial_value], dtype=torch.float32)
            else:  # Learnable bias with initial value
                self.logit_bias = torch.nn.Parameter(torch.ones(1) * self.bias_initial_value)
            logits += self.logit_bias

        self.adj_prob = logits

        for i in range(self.max_length):
            position = tf.ones([x.shape[0]]) * i
            position = tf.cast(position, tf.int32)

            # Update mask
            self.mask = tf.one_hot(position, self.max_length)

            masked_score = self.adj_prob[:,i,:] - 100000000.*self.mask
            prob = distr.Bernoulli(masked_score)    # probs input probability, logit input log_probability

            sampled_arr = prob.sample()    # Batch_size, seq_len for just one node

            self.samples.append(sampled_arr)
            self.mask_scores.append(masked_score)
            self.entropy.append(prob.entropy())

        return self.samples, self.mask_scores, self.entropy
