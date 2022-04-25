# https://github.com/arghosh/AKT/blob/master/akt.py
from posixpath import relpath
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Parameter, Embedding, Linear, ReLU, LayerNorm, Dropout, ModuleList, Softplus, Sequential, Sigmoid, BCEWithLogitsLoss
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import torch.nn.functional as F
import os
import yaml
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from enum import IntEnum
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

"""
AKT 구현 파라미터
         max_iter        300
         train_set       1
         seed    224
         optim   adam
         batch_size      24
         lr      1e-05
         maxgradnorm     -1
         final_fc_dim    512
         d_model         256
         d_ff    1024
         dropout         0.05
         n_block         1
         n_head          8
         kq_same         1
         l2      1e-05
         q_embed_dim     50
         qa_embed_dim    256
         memory_size     50
         init_std        0.1
         hidden_dim      512
         lamda_r         0.1
         lamda_w1        0.1
         lamda_w2        0.1
         model   akt_pid
         dataset         assist2017_pid
         seqlen          200
         data_dir        data/assist2017_pid
         data_name       assist2017_pid
         n_question      102
         n_pid   3162
         save    assist2017_pid
         load    assist2017_pid

"""
class AKT(Module):
    def __init__(self, num_skills, num_questions, seq_len, embedding_dim, num_blocks,
                 kq_same, model_type="akt", num_attn_heads=8, final_fc_dim=512, d_ff=2048, l2=1e-5,
                 dropout=0.2, separate_qr=False):
        super(AKT, self).__init__()

        """
        params:
            num_skills: # of skills
            num_questions: # of questions
            embedding_dim: embedding dim
            num_blocks: # of attn blocks
            seq_len: max length of sequenc
            kq_same: key랑 query랑 같은지
            num_attn_heads: number of heads if multi-headed attention
            final_fc_dim: dimension of final fully connected net before prediction
            d_ff: dimension for fully connected net inside the basic block
            
        """
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.seq_len = seq_len
        self.kq_same = kq_same
        print('kq_same', kq_same)
        self.model_type = model_type
        self.num_attn_heads = num_attn_heads
        self.final_fc_dim = final_fc_dim
        self.d_ff = d_ff
        self.l2 = l2
        self.dropout = dropout
        self.separate_qr = separate_qr

        if self.num_questions > 0:
            self.difficult_param = Embedding(self.num_questions, 1)  # /mu_{q_t} parameter
            self.q_embed_diff = Embedding(self.num_skills, self.embedding_dim)  # d_{c_t}
            self.qr_embed_diff = Embedding(2 * self.num_skills, self.embedding_dim)  # f_{(c_t, r_t)} or h_{r_t}
        self.q_embed = Embedding(2 * self.num_skills, self.embedding_dim)  # c_{c_t}
        self.qr_embed = Embedding(2 * self.num_skills, self.embedding_dim)  # e_{(c_t, r_t)}


        self.model = Architecture(
            n_question=self.num_skills,
            n_blocks=self.num_blocks,
            n_heads=self.num_attn_heads,
            dropout=self.dropout,
            d_model=self.embedding_dim,
            d_feature=self.embedding_dim/self.num_attn_heads,
            d_ff=self.d_ff,
            kq_same=self.kq_same,
            model_type=self.model_type
        )

        self.out = Sequential(
                Linear(2*self.embedding_dim, self.final_fc_dim),
                ReLU(),
                Dropout(self.dropout),
                Linear(self.final_fc_dim, self.final_fc_dim//2),
                ReLU(),
                Dropout(self.dropout),
                Linear(self.final_fc_dim//2, 1)
        )
        self.reset()
        self.loss_fn = nn.BCELoss(reduction='sum')
    
    def reset(self):
        for p in self.parameters():
            # question이 존재하며, difficult_param 일때! \mu_{q_t}
            if p.size(0) == self.num_questions+1 and self.num_questions > 0:
                torch.nn.init.constant_(p, 0.)

    def forward(self, feed_dict):
        q = feed_dict["skills"]
        r = feed_dict["responses"]
        masked_r = r * (r > -1).long()
        pid_data = feed_dict["questions"]

        # q: skill ids
        # qa: skill-response
        # t: true labels (response)...? 이걸 왜 굳이 여기에? train()함수에서 처리해도 될 듯?
        # pid_data => question ids
        qr = q + self.num_skills * masked_r
        
        q_embed_data = self.q_embed(q) # c_{c_t}: [batch_size, seq_len, embedding_dim]
        qr_embed_data = self.qr_embed(qr) # f_{(c_t, r_t)}: [batch_size, seq_len, d_model]


        if self.num_questions > 0:
            q_embed_diff_data = self.q_embed_diff(q) # d_{c_t}: variation vector
            pid_embed_data = self.difficult_param(pid_data) # \mu_{q_t}
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data # x_t = c_{c_t} + \mu_{q_t} + d_{c_t}
            qr_embed_diff_data = self.qr_embed_diff(qr) # f_{(c_t, r_t)} or h_{r_t}

            if self.separate_qr:
                qr_embed_data = qr_embed_data + pid_embed_data * qr_embed_diff_data
            else:
                # y_t = e_{(c_t, r_t)} + \mu_{q_t} * f_{(c_t, r_t)}
                # 이때 e_{(c_t, r_t)} = c_{c_t} + g_{r_t}
                # f_{(c_t, r_t)} = f_{(c_t, r_t)} + d_{c_t}
                # self.separate_qa == False하면, qa_data가 {0,1} response만을 포함하고 있어서
                # qa_embed_diff_data에는 response 정보만 있어서 q_embed_diff를 추가하는 듯?
                # 근데 q_embed_dff는 2Q+1 개의 voca를 갖는데.. 위에서 separate_qa=False이면 qa_data = {0,1}로 되는데?
                # 그냥 0,1만 쓰고 나머지는 안 쓰는 형태인 듯...?

                # e_{(c_t, r_t)} + \mu_{q_t} * (h_{r_t} + d_{c_t})
                qr_embed_data = qr_embed_data + pid_embed_data * (qr_embed_diff_data + q_embed_diff_data)               
            
            c_reg_loss = torch.sum(pid_embed_data**2.) * self.l2
        else:
            c_reg_loss = 0
        # [batch_size, seq_len, d_model]
        # pass to the decoder
        # output shape [batch_size, seq_len, d_model or d_model//2]
        # d_output is h_t
        d_output = self.model(q_embed_data, qr_embed_data)  # 211x512

        concat_q = torch.cat([d_output, q_embed_data], dim=-1) # concat([h_t, x_t])
        # attention block으로 부터 나온 d_output과 q_embed_data를 concat해서 out_layer에 넣어줌
        output = torch.sigmoid(self.out(concat_q)).squeeze()
        """
        AKT는 현재 (t) 스텝의 y정보를 masking함 (line 269)
        이를 확인하고자 다음과 같이 변경할시 label lekage가 발생해서 TEST AUC가 0.98로 치솟음
            'pred': output[:, 2:],
            'true': r[:, 1:-1].float(),
        
        따라서 r[:,1:]에 대한 알맞은 구현은 (t+1 step의 prediction에 대한) 다음이 맞는듯?
            'pred': output[:, 1:],
            'true': r[:, 1:].float(),        
        """
        out_dict = {
            'pred': output[:, 1:],
            'true': r[:, 1:].float(),
            'c_reg_loss': c_reg_loss
        }
        return out_dict
        
    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        c_reg_loss = out_dict["c_reg_loss"]
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        # print('token cnt', len(pred[mask]), 'label_sum', true[mask].sum().item())
        return loss + c_reg_loss, len(pred[mask]), true[mask].sum().item()


class Architecture(Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type
        print('model_type', model_type)

        if model_type == "akt":
            self.blocks_1 = ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks*2)
            ])

    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1:  # knowledge encoder: encode (question, response)'s
            # knowledge encoder
            # y^{\hat}_{t-1} = f_{enc_2} (y_1, ..., y_{t-1})
            # y는 current, past 둘다 봄
            """
            mask: 0 means that it can peek (엿보다) only past values.
            1 means that block can peek only current and past values
            """
            y = block(mask=1, query=y, key=y, values=y)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                # question encoder
                # x^{\hat}_{t} = f_{enc_1} (x_1, ..., x_t)
                # x는 current, past 둘다 봄
                """
                mask: 0 means that it can peek (엿보다) only past values.
                1 means that block can peek only current and past values
                """
                x = block(mask=1, query=x, key=x, values=x, apply_pos=False)
                flag_first = False
            else:  # dont peek current response
                # knoweldge retriever
                # h_t = f_{kr} (x^{\hat}_1, ..., x^{\hat}_t, y^{\hat}_1, ..., y^{\hat}_{t-1})
                # h는 past만 봄
                """
                mask: 0 means that it can peek (엿보다) only past values.
                1 means that block can peek only current and past values
                """
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
        return x



class TransformerLayer(Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super(TransformerLayer, self).__init__()
        """
            This is a Basic Block of Transformer paper.
            It contains one Multi-head attention object.
            Followed by layer norm and position-wise feed-forward net and dropotu layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same
        )

        # Two layer norm and two dropout layers
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)

        self.linear1 = Linear(d_model, d_ff)
        self.activation = ReLU()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_ff, d_model)

        self.layer_norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block: object of type BasicBlock(nn.Module). It contains maksed_attn_head objects which is of type MultiHeadAttnetion(nn.Module).
            mask: 0 means that it can peek (엿보다) only past values. 1 means that block can peek only current and past values
            query: Queries. In Transformer paper it is the input for both encoder and decoder
            key: Keys. In transformer paper it is the input for both encoder and decoder
            values: Values. In transformer paper it is the input for encoder and encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the alyer andr returned
        """

        batch_size, seqlen = query.size(0), query.size(1)
        """
        when mask==1
        >>> nopeek_mask (for question encoder, knoweldge encoder)
            array([[[[0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0]]]], dtype=uint8)

         >>> src_mask
            tensor([[[[ True, False, False, False, False],
                    [ True,  True, False, False, False],
                    [ True,  True,  True, False, False],
                    [ True,  True,  True,  True, False],
                    [ True,  True,  True,  True,  True]]]])

        when mask==0 (for knowledge retriever)
        >>> nopeek_mask
            array([[[[1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1]]]], dtype=uint8)

        >>> src_mask
            tensor([[[[False, False, False, False, False],
                    [ True, False, False, False, False],
                    [ True,  True, False, False, False],
                    [ True,  True,  True, False, False],
                    [ True,  True,  True,  True, False]]]])

        결과적으로 윗삼각원소들이 마스킹됨 (뒤에서 src_mask==0)
        row: target, col: source
        """
        device = query.get_device()
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')

        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)

        if mask == 0:  # If 0, zero-padding is needed
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))  # residual connection
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)

        return query




class MultiHeadAttention(Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super(MultiHeadAttention, self).__init__()
        """
        It has projection layer for getting keys, queries, and values. Followed by attention and a connected layer.
        """

        # d_feature=d_model // n_heads
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = Linear(d_model, d_model, bias=bias)
        self.k_linear = Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = Linear(d_model, d_model, bias=bias)
        self.dropout = Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = Linear(d_model, d_model, bias=bias)
        self.gammas = Parameter(torch.zeros(n_heads, 1, 1))
        xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)

        # perform linear operation and split into h heads

        # -1 부분이 seq_len인듯?, self.h = num_heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions batch_size * num_heads * seqlen * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k, mask,
                           self.dropout, zero_pad, gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        # concat torch.Size([24, 200, 256])   [batch_size, seqlen, d_model]
        # print('concat', concat.shape)
        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    """
    This is called by MultiHeadAttention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)  # [batch_size, 8, seq_len, seq_len]
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    # distnace 계산하기 위한 거네. 위치별로...
    """
    >>> x1 = torch.arange(5).expand(5, -1)
        tensor([[0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4]])

    >>> x2 = x1.transpose(0, 1).contiguous()
        tensor([[0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4]])
    """
    x1 = torch.arange(seqlen).expand(seqlen, -1)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        # 0인 부분을 mask, src_mask에서 False인 부분 mask, 즉 윗삼각원소들 마스킹, 행이 target, 열이 source
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # [batch_size, 8, seqlen, seqlen]
        scores_ = scores_ * mask.float()

        # d(t, \tau)의 \gamma_{t, t'}에서 분자, 분모?
        # [batch_size, 8, seqlen, seqlen]
        distcum_scores = torch.cumsum(scores_, dim=-1)
        # [batch_size, 8, seqlen, 1]
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        """
        >>> x1-x2
            tensor([[ 0,  1,  2,  3,  4],
                    [-1,  0,  1,  2,  3],
                    [-2, -1,  0,  1,  2],
                    [-3, -2, -1,  0,  1],
                    [-4, -3, -2, -1,  0]])

        >>> torch.abs(x1-x2)
            tensor([[0, 1, 2, 3, 4],
                    [1, 0, 1, 2, 3],
                    [2, 1, 0, 1, 2],
                    [3, 2, 1, 0, 1],
                    [4, 3, 2, 1, 0]])
        """
        device = distcum_scores.get_device()
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor)  # [1, 1, seqlen, seqlen]
        position_effect = position_effect.to(device)
        # [batch_size, 8, seqlen, seqlen] positive distance
        # dist_score => d(t, tau)
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores) * position_effect, min=0.
        )
        dist_scores = dist_scores.sqrt().detach()

    m = Softplus()
    # 1,8,1,1  gamma is \theta in the paper (learnable decay rate parameter)
    gamma = -1. * m(gamma).unsqueeze(0)
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e-5
    total_effect = torch.clamp(torch.clamp(
        (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # [batch_size, 8, seq_len, seq_len]
    """
    zero_pad가 필요한가?
    어차피 데이터 전처리에서 [1:] [:-1] slicing을 통해서 처리해줬으므로, 따로 첫번째 token에 대한 padding은 필요 없을듯?
    참고로, nopeek mask (when mask=0)을 통해서 과거 step만 보도록 처리된다고 생각하면 됨 (by zeroing out)
    """
    # if zero_pad:
    #     pad_zero = torch.zeros(bs, head, 1, seqlen)
    #     scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class LearnablePositionalEmbedding(Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(torch.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

