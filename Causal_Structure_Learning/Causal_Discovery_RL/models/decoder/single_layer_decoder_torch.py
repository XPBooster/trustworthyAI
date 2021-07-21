import tensorflow as tf
# from tensorflow.contrib import distributions as distr
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F

class SingleLayerDecoder(nn.Module):

    def __init__(self, config, is_train=True):

        super(SingleLayerDecoder, self).__init__()
        self.num_nodes = config.num_nodes    # input sequence length (number of cities)
        self.d_model = config.d_model    # dimension of embedding space (actor)
        self.decoder_d_model = config.decoder_d_model
        # self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer
        self.decoder_activation = config.decoder_activation
        self.use_bias = config.use_bias
        self.bias_initial_value = config.bias_initial_value
        self.use_bias_constant = config.use_bias_constant

        self.is_training = is_train

    def forward(self, x):

        sample_list, mask_score_list, entropy_list = [], [], []
        mask = 0
        # encoder_output is a tensor of size [batch_size, num_nodes, input_embed]
        # with tf.variable_scope('singe_layer_nn'):
        W_l = torch.nn.Parameter(nn.init.xavier_normal_(torch.empty(self.d_model, self.decoder_d_model)))
        W_r = torch.nn.Parameter(nn.init.xavier_normal_(torch.empty(self.d_model, self.decoder_d_model)))
        W_u = torch.nn.Parameter(nn.init.kaiming_normal_(torch.empty(1, self.decoder_d_model)).squeeze())    # Aggregate across decoder hidden dim


        dot_l = torch.einsum('ijk, kl->ijl', x, W_l)
        dot_r = torch.einsum('ijk, kl->ijl', x, W_r)

        tiled_l = torch.unsqueeze(dot_l, dim=2)
        tiled_l = tiled_l.expand(-1, -1, self.num_nodes, -1)
        tiled_r = torch.unsqueeze(dot_r, dim=1)
        tiled_r = tiled_r.expand(-1, self.num_nodes, -1, -1)
        if self.decoder_activation == 'tanh':    # Original implementation by paper
            final_sum = torch.tanh(tiled_l + tiled_r)
        elif self.decoder_activation == 'relu':
            final_sum = torch.relu(tiled_l + tiled_r)
        elif self.decoder_activation == 'none':    # Without activation function
            final_sum = tiled_l + tiled_r
        else:
            raise NotImplementedError('Current decoder activation is not implemented yet')

        # final_sum is of shape (batch_size, num_nodes, num_nodes, decoder_d_model)
        logits = torch.einsum('ijkl, l->ijk', final_sum, W_u)    # batch, max_len, max_len

        if self.use_bias:    # Bias to control sparsity/density
            if self.bias_initial_value is None:  # Randomly initialize the learnable bias
                self.logit_bias = torch.nn.Parameter(nn.init.xavier_normal_(torch.empty(1, 1)).squeeze())
            elif self.use_bias_constant:  # Constant bias
                self.logit_bias = torch.tensor([self.bias_initial_value], dtype=torch.float64)
            else:  # Learnable bias with initial value
                self.logit_bias = torch.nn.Parameter(torch.ones(1) * self.bias_initial_value)
            logits += self.logit_bias

        self.adj_prob = logits

        for i in range(self.num_nodes):

            position = torch.ones([x.shape[0]]) * i
            position = position.type(torch.int32)

            # Update mask
            self.mask = torch.zeros([x.shape[0], self.num_nodes])
            self.mask[:,i] = -10000000
            # torch.nn.functional.one_hot(position, self.num_nodes)
            masked_score = self.adj_prob[:,i,:] + self.mask
            prob = Bernoulli(torch.sigmoid(masked_score))    # probs input probability, logit input log_probability
            # prob = F.relu(prob)
            sampled_arr = prob.sample()    # Batch_size, seq_len for just one node

            sample_list.append(sampled_arr)
            mask_score_list.append(masked_score)
            entropy_list.append(prob.entropy())

        return sample_list, mask_score_list, entropy_list
