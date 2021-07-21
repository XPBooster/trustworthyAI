# -*- coding: utf-8 -*-

import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):

    # Scaled Dot-Product Attention

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_hidden, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.w_qs = nn.Linear(d_model, n_head * d_hidden)
        self.w_ks = nn.Linear(d_model, n_head * d_hidden)
        self.w_vs = nn.Linear(d_model, n_head * d_hidden)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_hidden)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_hidden)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_hidden)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_hidden, 0.2))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_hidden, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, self.n_head, self.d_hidden)
        k = self.w_ks(k).view(sz_b, len_k, self.n_head, self.d_hidden)
        v = self.w_vs(v).view(sz_b, len_v, self.n_head, self.d_hidden)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_hidden)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_hidden)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.d_hidden)  # (n*b) x lv x dv
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(self.n_head, sz_b, len_q, self.d_hidden)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_model, dropout=0.1):

        super().__init__()
        self.w_1 = nn.Conv1d(d_model, d_model, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        output = x.transpose(2, 1)
        output = F.relu(self.w_1(output))
        output = output.transpose(2, 1)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_hidden, n_head):

        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(n_head, d_model, d_hidden)
        self.ffn = PositionwiseFeedForward(d_model)

    def forward(self, x, mask):

        output, _ = self.self_attn(x, x, x, mask=None)
        output = self.ffn(output)

        return output


class TransformerEncoder(nn.Module):

    def __init__(self, config, is_train=True):

        super(TransformerEncoder, self).__init__()
        block = EncoderLayer(config.d_model, config.d_model_attn, config.num_heads)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(config.num_stacks)])
        self.norm = nn.LayerNorm(config.d_model)
        self.w_1 = nn.Conv1d(config.input_dimension, config.d_model, 1)  # position-wise

    def forward(self, x, mask=None):

        x = x.type(dtype=torch.float32)
        output = x.transpose(2, 1)
        output = self.w_1(output)
        output = output.transpose(2, 1)
        for block in self.blocks:
            output = block(output, mask)
        output = self.norm(output)

        return output