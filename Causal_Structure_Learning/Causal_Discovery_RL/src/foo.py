'''
@Project ：trustworthyAI_torch 
@File    ：foo.py
@Author  ：吾非同
@Date    ：2021/7/18 22:09 
'''

import torch
from torch.utils.data import DataLoader
from data_loader.dataset_read_data import CausalDataSet
from models.encoder.transformer_encoder_torch import TransformerEncoder
from helpers.config_graph import get_config, print_config
model = TransformerEncoder(n_stack=2, d_in=64, d_model=32, d_hidden=16, n_head=8)

config, _ = get_config()
file_path = f'{config.data_path}/real_data'
solution_path = f'{config.data_path}/true_graph'
training_set = CausalDataSet(config)
training_loader = DataLoader(dataset=training_set, batch_size=config.batch_size, shuffle=True)
for batch_x in training_loader:
    out = model(batch_x)
    print(out)