import pandas as pd
import numpy as np
import torch
import os
from torch._C import device
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
import yaml
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
from utils import EarlyStopping

class DKT(Module):
    def __init__(self, num_questions, embedding_dim, hidden_dim, num_layers, dropout):
        super(DKT, self).__init__()
        self.num_questions = num_questions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.interaction_emb = Embedding(
            self.num_questions * 2, self.embedding_dim)
        self.lstm = LSTM(
            self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True
        )
        self.out_layer = Linear(
            self.hidden_dim, self.num_questions)
        self.dropout = Dropout(dropout)
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, feed_dict):
        q = feed_dict["skills"]
        r = feed_dict["responses"]

        masked_r = r * (r > -1).long()
        # print('q', q[10,:])
        # print('r', r[10,:])
        qr = q + self.num_questions * masked_r
        self.lstm.flatten_parameters()
        h, _ = self.lstm(self.interaction_emb(qr))
        y = self.out_layer(h)
        y = self.dropout(y)
        pred = torch.sigmoid(y)
        # print('pred', pred[10,:])
        """
        length 199, 200 차이... target_question은 199이고

        pred는 200...

        참고: https://github.com/marshallgrimmett/AKT/blob/master/dkt.py
        참고: Hawkes KT --> DKT
        
        embed_history_i_packed = torch.nn.utils.rnn.pack_padded_sequence(embed_history_i, lengths.cpu() - 1, batch_first=True)
        
        pack_padded_seq, pad_pakced_seq 활용해서 length-1로 만들고.. gather하는 거 같음...
        """
        target_question = q[:, 1:].unsqueeze(-1) # for next question prediction
        pred = pred[:, :-1].gather(-1, target_question).squeeze(-1) # one-hot encoding for the target question
        # 마지막스텝 (T)은, 이미 인풋에 q_T, r_T, 예측못하지? Y_T, TRUE_(T+1)
        # Y * \delta(q), T
        
        
        out_dict = {
            'pred': pred,
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

