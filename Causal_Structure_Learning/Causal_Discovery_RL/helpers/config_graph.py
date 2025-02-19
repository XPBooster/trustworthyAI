#-*- coding: utf-8 -*-
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


net_arg = add_argument_group('Network')
# Encoder
net_arg.add_argument('--encoder_type', type=str, default='TransformerEncoder', help='type of encoder used')
net_arg.add_argument('--d_model', type=int, default=32, help='actor neural num')
net_arg.add_argument('--d_model_attn', type=int, default=16, help='actor neural num in self-attention')
net_arg.add_argument('--num_heads', type=int, default=8, help='actor input embedding')
net_arg.add_argument('--num_stacks', type=int, default=6, help='actor LSTM num_neurons')
net_arg.add_argument('--residual', type=bool, default=True, help='whether to use residual for gat encoder')

# Decoder
net_arg.add_argument('--decoder_type', type=str, default='SingleLayerDecoder', help='type of decoder used')
net_arg.add_argument('--decoder_activation', type=str, default='tanh', help='activation for decoder')    # Choose from: 'tanh', 'relu', 'none'
net_arg.add_argument('--decoder_d_model', type=int, default=16, help='hidden dimension for decoder')
net_arg.add_argument('--use_bias', type=bool, default=True, help='Whether to add bias term when calculating decoder logits')
net_arg.add_argument('--use_bias_constant', type=bool, default=True, help='Whether to add bias term as CONSTANT when calculating decoder logits')
net_arg.add_argument('--bias_initial_value', type=float, default=False, help='Initial value for bias term when calculating decoder logits')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default=512, help='batch size for training')
data_arg.add_argument('--input_dimension', type=int, default=64, help='dimension of reshaped vector') # reshaped
data_arg.add_argument('--num_nodes', type=int, default=55, help='number of variables')
data_arg.add_argument('--data_size', type=int, default=3000, help='Number of observational samples')

data_arg.add_argument('--read_data', type=bool, default=True, help='read existing_data or not')
data_arg.add_argument('--data_path', type=str, default='./Datasets/real_data.csv', help='data path for read data')
data_arg.add_argument('--graph_path', type=str, default='Datasets/real_graph.csv', help='data path for read graph')
data_arg.add_argument('--normalize', type=bool, default=True, help='whether the inputdata shall be normalized')
data_arg.add_argument('--transpose', type=bool, default=True, help='whether the true graph needs transposed')

data_arg.add_argument('--score_type', type=str, default='BIC', help='score functions')
data_arg.add_argument('--reg_type', type=str, default='LR', help='regressor type (in combination wth score_type)')
data_arg.add_argument('--lambda_iter_num', type=int, default=1000, help='how often to update lambdas')
#TODO: maybe add other weights adjustment strategy as an option
data_arg.add_argument('--lambda_flag_default', type=bool, default=True, help='with set lambda parameters; true with default strategy and ignore input bounds')
data_arg.add_argument('--score_bd_tight', action='store_true', help='if bound is tight, then simply use a fixed value, rather than the adaptive one')
data_arg.add_argument('--lambda1_update', type=float, default=1, help='increasing additive lambda1')
data_arg.add_argument('--lambda2_update', type=float, default=10, help='increasing  multiplying lambda2')
data_arg.add_argument('--score_lower', type=float, default=0.0, help='lower bound on lambda1')
data_arg.add_argument('--score_upper', type=float, default=0.0, help='upper bound on lambda1')
data_arg.add_argument('--lambda2_lower', type=float, default=-1, help='lower bound on lambda2')
data_arg.add_argument('--lambda2_upper', type=float, default=-1, help='upper bound on lambda2')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--seed', type=int, default=8, help='seed')
train_arg.add_argument('--nb_epoch', type=int, default=20000, help='nb epoch')
train_arg.add_argument('--lr1', type=float, default=0.001, help='actor learning rate')
train_arg.add_argument('--lr1_tinit', type=float, default=4, help='the period of the first epoch of lr1')
train_arg.add_argument('--lr1_tmult', type=float, default=2, help='the multiplier of the period of lr1')
train_arg.add_argument('--lr2', type=float, default=0.001, help='actor learning rate')
train_arg.add_argument('--lr2_tinit', type=float, default=4, help='the period of the first epoch of lr1')
train_arg.add_argument('--lr2_tmult', type=float, default=2, help='the multiplier of the period of lr1')
train_arg.add_argument('--alpha', type=float, default=0.99, help='update factor moving average baseline')
train_arg.add_argument('--init_baseline', type=float, default=-1.0, help='initial baseline - REINFORCE')
train_arg.add_argument('--temperature', type=float, default=3.0, help='pointer_net initial temperature')
train_arg.add_argument('--C', type=float, default=10.0, help='pointer_net tan clipping')
train_arg.add_argument('--l1_graph_reg', type=float, default=0.0, help='L1 graph regularization to encourage sparsity')

# Misc
misc_arg = add_argument_group('User options') #####################################################

misc_arg.add_argument('--inference_mode', type=str2bool, default=True, help='switch to inference mode when model is trained')
misc_arg.add_argument('--restore_model', type=str2bool, default=False, help='whether or not model is retrieved')

misc_arg.add_argument('--save_to', type=str, default='20/model', help='saver sub directory')
misc_arg.add_argument('--restore_from', type=str, default='20/model', help='loader sub directory')  ###
misc_arg.add_argument('--log_dir', type=str, default='summary/20/repo', help='summary writer log directory') 
misc_arg.add_argument('--verbose', type=bool, default=False, help='print detailed logging or not')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_config():
    config, _ = get_config()
    print('\n')
    print('Data Config:')
    print('* Batch size:',config.batch_size)
    print('* Sequence length:',config.num_nodes)
    print('* City coordinates:',config.input_dimension)
    print('\n')
    print('Network Config:')
    print('* Restored model:',config.restore_model)
    print('* Actor hidden_dim (embed / num neurons):',config.hidden_dim)
    print('* Actor tan clipping:',config.C)
    print('\n')
    if config.inference_mode==False:
        print('Training Config:')
        print('* Nb epoch:',config.nb_epoch)
        print('* Temperature:',config.temperature)
        print('* Actor learning rate (init,decay_step,decay_rate):',config.lr1_start,config.lr1_decay_step,config.lr1_decay_rate)
    else:
        print('Testing Config:')
        print('* Summary writer log dir:',config.log_dir)
        print('\n')
