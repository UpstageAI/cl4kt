import pandas as pd
import numpy as np
import torch
import os

from torch.nn import Module, Parameter, Embedding, Linear
from torch.nn.init import xavier_normal_

from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class DKVMN(Module):
    def __init__(self, num_q, key_dim, value_dim, num_concepts):
        super(DKVMN, self).__init__()
        self.num_q = num_q
        self.dim_k = key_dim
        self.dim_v = value_dim
        self.num_concepts = num_concepts

        self.k_emb_layer = Embedding(self.num_q, self.dim_k)
        self.Mk = Parameter(torch.Tensor(self.dim_k, self.num_concepts))
        self.Mv = Parameter(torch.Tensor(self.num_concepts, self.dim_v))

        xavier_normal_(self.Mk)
        xavier_normal_(self.Mv)

        self.v_emb_layer = Embedding(self.num_q * 2, self.dim_v)

        self.f_layer = Linear(self.dim_k * 2, self.dim_k)
        self.p_layer = Linear(self.dim_k, 1)

        self.e_layer = Linear(self.dim_v, self.dim_v)
        self.a_layer = Linear(self.dim_v, self.dim_v)

        self.loss_fn = torch.nn.BCELoss()

    def forward(self, feed_dict):
        q = feed_dict["skills"]
        r = feed_dict["responses"]

        masked_r = r * (r > -1).long()
        qr = q + self.num_q * masked_r
        Mvt = self.Mv.unsqueeze(0)

        p = []
        Mv = []

        for qt, qrt in zip(q.permute(1, 0), qr.permute(1, 0)):
            kt = self.k_emb_layer(qt)
            vt = self.v_emb_layer(qrt)

            wt = torch.softmax(torch.matmul(kt, self.Mk), dim=-1)

            # Read Process
            rt = (wt.unsqueeze(-1) * Mvt).sum(1)
            ft = torch.tanh(self.f_layer(torch.cat([rt, kt], dim=-1)))
            pt = torch.sigmoid(self.p_layer(ft)).squeeze()

            # Write Process
            et = torch.sigmoid(self.e_layer(vt))
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1)))
            at = torch.tanh(self.a_layer(vt))
            Mvt = Mvt + (wt.unsqueeze(-1) * at.unsqueeze(1))

            p.append(pt)
            Mv.append(Mvt)
            # print('pt', pt.shape)
        p = torch.stack(p, dim=1)
        Mv = torch.stack(Mv, dim=1)

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
        
        return loss
