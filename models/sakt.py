# https://github.com/hcnoh/knowledge-tracing-collection-pytorch/blob/main/models/sakt.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Parameter, Embedding
from torch.nn.init import xavier_normal_
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class SAKT(Module):
    def __init__(self, num_questions, seq_len, embedding_dim, num_layers, num_attn_heads, dropout):
        super(SAKT, self).__init__()
        self.num_questions = num_questions
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout

        self.M = Embedding(self.num_questions * 2, self.embedding_dim)
        self.E = Embedding(self.num_questions, self.embedding_dim)
        self.P = Parameter(torch.Tensor(self.seq_len, self.embedding_dim))
        xavier_normal_(self.P)

        self.attn = nn.MultiheadAttention(
            self.embedding_dim, self.num_attn_heads, dropout=self.dropout
        )
        self.attn_dropout = nn.Dropout(self.dropout)
        self.attn_layer_norm = nn.LayerNorm([self.embedding_dim])

        self.FFN = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Dropout(self.dropout),
        )
        self.FFN_layer_norm = nn.LayerNorm([self.embedding_dim])

        self.pred = nn.Linear(self.embedding_dim, 1)

        self.loss_fn = nn.BCELoss(reduction='sum')

    def forward(self, feed_dict):
        q = feed_dict["skills"]
        # print('q', q.shape)
        pad_zero = torch.zeros((q.shape[0], 1)).long().cuda()
        q_shifted = torch.cat([q[:, 1:], pad_zero], dim=1)
        # print('q_shifted', q_shifted.shape)
        r = feed_dict["responses"]
        masked_r = r * (r > -1).long()

        x = q + self.num_questions * masked_r

        M = self.M(x).permute(1, 0, 2)
        E = self.E(q_shifted).permute(1, 0, 2)
        P = self.P.unsqueeze(1)

        causal_mask = torch.triu(
            torch.ones([E.shape[0], M.shape[0]]), diagonal=1
        ).bool()

        M += P

        S, attn_weights = self.attn(E, M, M, attn_mask=causal_mask)

        # print('S', S.shape) # [L, B, D]
        # print('attn_weights', attn_weights.shape) # [B, L, L]
        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)

        S = self.attn_layer_norm(S + M + E)

        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        p = torch.sigmoid(self.pred(F)).squeeze()

        out_dict = {
            'pred': p[:, :-1],
            'true': r[:, 1:].float()
        }
        return out_dict

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        # print('token cnt', len(pred[mask]), 'label_sum', true[mask].sum().item())
        return loss, len(pred[mask]), true[mask].sum().item()

