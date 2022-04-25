# https://github.com/arghosh/AKT/blob/master/akt.py
import math
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Parameter, Embedding, Linear, ReLU, Dropout, ModuleList, Softplus, Sequential, Sigmoid, BCEWithLogitsLoss
import torch.nn.functional as F


if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
"""
SimCLR 이해하기 쉽게 설명
https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/

"""
class CL4KT(Module):
    def __init__(self, num_skills, num_questions, seq_len, hidden_dim, num_blocks, num_attn_heads, dropout, reg_cl,
                 q_mask_prob, s_mask_prob, r_mask_prob):
        super(CL4KT, self).__init__()
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.num_attn_heads = num_attn_heads
        self.reg_cl = reg_cl

        self.question_mask_idx = self.num_questions
        self.skill_mask_idx = self.num_skills
        self.response_mask_idx = 2
        self.interaction_mask_idx = 2 * self.num_skills + 1 # 0 for padding, 2*self.num_skills+1 for masking

        self.question_embed = Embedding(self.num_skills+1, self.hidden_dim, padding_idx=0)
        self.interaction_embed = Embedding(2 * self.num_skills + 2, self.hidden_dim, padding_idx=0) # 2 * (num_skills+1_) + 1?
        self.position_embeddings = Embedding(seq_len, self.hidden_dim)

        self.question_encoder = Encoder(hidden_dim, num_blocks, num_attn_heads, dropout)
        self.interaction_encoder = Encoder(hidden_dim, num_blocks, num_attn_heads, dropout)

        self.LayerNorm = LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.out_norm = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.apply(self.init_weights)

        self.loss_fn = nn.BCELoss(reduction='sum')

    # https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py
    def nt_xent_loss(self, out_1, out_2, temperature=0.1, eps=1e-6):
        """
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """

        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        out_1 = F.normalize(out_1, dim=-1)
        out_2 = F.normalize(out_2, dim=-1)
        out_1_dist = out_1
        out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        # print('out', out.shape)
        # print('out_dist', out_dist.shape)


        cov = torch.matmul(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = torch.tensor(neg.shape).fill_(math.e**(1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss



    # def interaction_interaction_prediction(self, sequence_output, target_interaction):
    #     sequence_output = self.qr_qr_norm(sequence_output.view([-1, self.hidden_dim]))
    #     target_interaction = target_interaction.view([-1, self.hidden_dim])
    #     score = torch.mul(sequence_output, target_interaction)
    #     return torch.sigmoid(torch.sum(score, -1))

    # def question_question_prediction(self, sequence_output, target_question):
    #     sequence_output = self.q_q_norm(sequence_output.view([-1, self.hidden_dim]))
    #     target_question = target_question.view([-1, self.hidden_dim])
    #     score = torch.mul(sequence_output, target_question)
    #     return torch.sigmoid(torch.sum(score, -1))

    # def question_interaction_prediction(self, sequence_output, target_interaction):
    #     sequence_output = self.q_qr_norm(sequence_output.view([-1, self.hidden_dim]))
    #     target_interaction = target_interaction.view([-1, self.hidden_dim])
    #     score = torch.mul(sequence_output, target_interaction)
    #     return torch.sigmoid(torch.sum(score, -1))

    def add_position_embedding(self, sequence_emb, position_emb):
        sequence_emb = sequence_emb + position_emb
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)        
        
        return sequence_emb

    def question_embeddings(self, questions):
        question_embs = self.question_embed(questions)
        question_masks = (questions == 0).float() * -1e8
        question_masks = torch.unsqueeze(torch.unsqueeze(question_masks, 1), 1)
        return question_embs, question_masks

    def interaction_embeddings(self, questions, responses):
        q_masks = (questions==self.skill_mask_idx).bool().to(questions.device)
        r_masks = (responses==self.response_mask_idx).bool().to(questions.device)
        masked_responses = responses * (responses > -1).long()
        interactions = questions + self.num_skills * masked_responses
        masks = q_masks | r_masks
        interactions[masks] = self.interaction_mask_idx
        # interaction_mask_idx가 226인데 interaction index에 336이 나옴
        # print('mask', masks)
        # print('questions', questions)
        # print('num_skills', self.num_skills)
        # print('masked_responses', masked_responses)
        # print('interations', interactions, self.interaction_mask_idx)
        
        # interactions[r_masks] = self.interaction_mask_idx
        # print(masks)
        # print('a', interactions, 2 * self.num_skills + 1, )
        # assert torch.sum((questions > self.num_skills+1).long()) > 1
        
        # print(interactions, self.interaction_mask_idx, )
        """
        q 혹은 r 이 masking된 경우엔 interaction mask idx로 대체
        """
        # print('q_mask', q_masks.shape)
        # print('r_mask', r_masks.shape)
        # print('interactions', interactions.shape)



        position_ids = torch.arange(self.seq_len, dtype=torch.long, device=interactions.device)
        position_ids = position_ids.unsqueeze(0).expand_as(interactions)
        position_embeddings = self.position_embeddings(position_ids)

        interaction_embs = self.add_position_embedding(self.interaction_embed(interactions), position_embeddings)
        interactions_masks = (interactions == 0).float() * -1e8
        interactions_masks = torch.unsqueeze(torch.unsqueeze(interactions_masks, 1), 1)

        return interaction_embs, interactions_masks

    def get_question_score(self, questions):
        question_embs, question_masks = self.question_embeddings(questions)
        question_encoded_layers = self.question_encoder(question_embs, question_masks, output_all_encoded_layers=True)
        question_encoded_output = question_encoded_layers[-1]
        # print('question_encoded_output', question_encoded_output.shape)
        return question_encoded_output

    def get_interaction_score(self, questions, responses):
        interaction_embs, interaction_masks = self.interaction_embeddings(questions, responses)
        interaction_encoded_layers = self.interaction_encoder(interaction_embs, interaction_masks, output_all_encoded_layers=True)
        interaction_encoded_output = interaction_encoded_layers[-1]
        
        return interaction_encoded_output

    def prediction(self, batch):
        q = batch["skills"] # augmented q_i, augmented q_j and original q
        r = batch["responses"] # augmented r_i, augmented r_j and original r
        
        pad_zero = torch.zeros((q.shape[0], 1)).long().to(q.device)
        q_shifted = torch.cat([q[:, 1:], pad_zero], dim=1) # SAKT-like prediciton...

 
        question_score = self.get_question_score(q_shifted)

        interaction_score = self.get_interaction_score(q, r)

        context = self.out_norm(interaction_score)
        score = torch.mul(context, question_score)
        score = torch.sigmoid(torch.sum(score, dim=-1))


        out_dict = {
            'pred': score[:, :-1],
            'true': r[:, 1:].float()
        }

        return out_dict        

    def forward(self, batch):
        if self.training:
            q_i, q_j, q = batch["skills"] # augmented q_i, augmented q_j and original q
            r_i, r_j, r = batch["responses"] # augmented r_i, augmented r_j and original r
            
            pad_zero = torch.zeros((q.shape[0], 1)).long().to(q.device)
            q_shifted = torch.cat([q[:, 1:], pad_zero], dim=1) # SAKT-like prediciton...

            # print('q', q[10,:])
            # print('r', r[10,:])
            question_i_score = self.get_question_score(q_i)
            question_j_score = self.get_question_score(q_j)
            question_score = self.get_question_score(q_shifted)

            interaction_i_score = self.get_interaction_score(q_i, r_i)
            interaction_j_score = self.get_interaction_score(q_j, r_j)
            interaction_score = self.get_interaction_score(q_shifted, r) # or q?

            """
            Sequence Encoder의 last step hidden state 활용, projection 도 해주기
            """
            # question sim
            question_loss = self.nt_xent_loss(question_i_score[:,-1,:], question_j_score[:,-1,:])
            # interaction sim
            interaction_loss = self.nt_xent_loss(interaction_i_score[:,-1,:], interaction_j_score[:,-1,:])
            # question-intearction sim
            question_interaction_loss = self.nt_xent_loss(question_j_score[:,-1,:], interaction_j_score[:,-1,:])

            context = self.out_norm(interaction_score)
            score = torch.mul(context, question_score)
            score = torch.sigmoid(torch.sum(score, dim=-1))


            out_dict = {
                'pred': score[:, :-1],
                'true': r[:, 1:].float(),
                'cl_loss': question_loss + interaction_loss + question_interaction_loss
            }
        else:
            q = batch["skills"] # augmented q_i, augmented q_j and original q
            r = batch["responses"] # augmented r_i, augmented r_j and original r
            
            pad_zero = torch.zeros((q.shape[0], 1)).long().to(q.device)
            q_shifted = torch.cat([q[:, 1:], pad_zero], dim=1) # SAKT-like prediciton

            question_score = self.get_question_score(q_shifted)

            interaction_score = self.get_interaction_score(q, r)

            context = self.out_norm(interaction_score)
            score = torch.mul(context, question_score)
            score = torch.sigmoid(torch.sum(score, dim=-1))

            out_dict = {
                'pred': score[:, :-1],
                'true': r[:, 1:].float()
            }

        return out_dict


    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()        

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        cl_loss = out_dict["cl_loss"]
        mask = true > -1

        # print('pred', pred)
        # print('true', true.shape)
        # print('mask', mask)
        # print('pred[mask]', pred[mask])
        loss = self.loss_fn(pred[mask], true[mask]) + self.reg_cl * cl_loss
        # print('token cnt', len(pred[mask]), 'label_sum', true[mask].sum().item())
        return loss, len(pred[mask]), true[mask].sum().item()

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_attn_heads, dropout):
        super(SelfAttention, self).__init__()

        if hidden_dim % num_attn_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, num_attn_heads))
        self.num_attn_heads = num_attn_heads
        self.attention_head_size = int(hidden_dim / num_attn_heads)
        self.all_head_size = self.num_attn_heads * self.attention_head_size

        # print('hidden_size', hidden_dim)
        # print('attention_head_size', self.attention_head_size)
        # print('num_attn_heads', self.num_attn_heads)
        # print('all_head_size', self.all_head_size)
        self.query = nn.Linear(hidden_dim, self.all_head_size)
        self.key = nn.Linear(hidden_dim, self.all_head_size)
        self.value = nn.Linear(hidden_dim, self.all_head_size)

        self.attn_dropout = nn.Dropout(dropout)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.LayerNorm = LayerNorm(hidden_dim, eps=1e-12)
        self.out_dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attn_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # attention_scores: [batch_size heads seq_len seq_len] scores
        # attention_mask: [batch_size 1 seq_len seq_len]
        # print('attention_scores', attention_scores.shape)
        # print('attention_mask', attention_mask.shape)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        # print('attention_probs', attention_probs.shape)
        # print('value_layer', value_layer.shape)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Intermediate(nn.Module):
    def __init__(self, hidden_dim, num_blocks, num_attn_heads, dropout):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.intermediate_act_fn = gelu
        # if isinstance(args.hidden_act, str):
        #     self.intermediate_act_fn = ACT2FN[args.hidden_act]
        # else:
        #     self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.LayerNorm = LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)


    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Layer(nn.Module):
    def __init__(self, hidden_dim, num_blocks, num_attn_heads, dropout):
        super(Layer, self).__init__()
        self.attention = SelfAttention(hidden_dim, num_attn_heads, dropout)
        self.intermediate = Intermediate(hidden_dim, num_blocks, num_attn_heads, dropout)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output

class Encoder(nn.Module):
    def __init__(self, hidden_dim, num_blocks, num_attn_heads, dropout):
        super(Encoder, self).__init__()
        layer = Layer(hidden_dim, num_blocks, num_attn_heads, dropout)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(num_blocks)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers