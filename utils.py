import numpy as np
from numpy.core.fromnumeric import shape
from torch.nn.utils.rnn import pad_sequence
from torch import FloatTensor, LongTensor
import torch
import random

def augment_kt_seqs(q_seq, s_seq, r_seq, q_mask_prob, s_mask_prob, r_mask_prob,
                    q_mask_id, s_mask_id, r_mask_id, seed=None, skill_rel=None):
    # masking (randoml or PMI 등을 활용해서)
    # 구글 논문의 Correlated Feature Masking 등...
    random.seed(seed)
    masked_q_seq = []
    masked_s_seq = []
    masked_r_seq = []

    for i, q in enumerate(q_seq):
        prob = random.random()
        if prob < q_mask_prob and q != 0:
            masked_q_seq.append(q_mask_id)
        else:
            masked_q_seq.append(q)
        prob = random.random()
        if prob < s_mask_prob and q != 0:
            masked_s_seq.append(s_mask_id)
        else:
            masked_s_seq.append(s_seq[i])
        prob = random.random()
        if prob < r_mask_prob and q != 0:
            masked_r_seq.append(r_mask_id)
        else:
            masked_r_seq.append(r_seq[i])

    return masked_q_seq, masked_s_seq, masked_r_seq
    

# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def preprocess_qr(questions, responses, seq_len, pad_val=-1):
    """
    split the interactions whose length is more than seq_len
    """
    preprocessed_questions = []
    preprocessed_responses = []

    for q, r in zip(questions, responses):
        i = 0
        while i + seq_len < len(q):
            preprocessed_questions.append(q[i:i + seq_len])
            preprocessed_responses.append(r[i:i + seq_len])

            i += seq_len

        preprocessed_questions.append(
            np.concatenate(
                [
                    q[i:],
                    np.array([pad_val] * (i + seq_len - len(q)))
                ]
            )
        )
        preprocessed_responses.append(
            np.concatenate(
                [
                    r[i:],
                    np.array([pad_val] * (i + seq_len - len(q)))
                ]
            )
        )

    return preprocessed_questions, preprocessed_responses

def preprocess_qsr(questions, skills, responses, seq_len, pad_val=0):
    """
    split the interactions whose length is more than seq_len
    """
    preprocessed_questions = []
    preprocessed_skills = []
    preprocessed_responses = []

    for q, s, r in zip(questions, skills, responses):
        i = 0
        while i + seq_len < len(q):
            preprocessed_questions.append(q[i:i + seq_len])
            preprocessed_skills.append(s[i:i + seq_len])
            preprocessed_responses.append(r[i:i + seq_len])

            i += seq_len

        preprocessed_questions.append(
            np.concatenate(
                [
                    q[i:],
                    np.array([pad_val] * (i + seq_len - len(q)))
                ]
            )
        )
        preprocessed_skills.append(
            np.concatenate(
                [
                    s[i:],
                    np.array([pad_val] * (i + seq_len - len(q)))
                ]
            )
        )
        preprocessed_responses.append(
            np.concatenate(
                [
                    r[i:],
                    np.array([-1] * (i + seq_len - len(q)))
                ]
            )
        )

    return preprocessed_questions, preprocessed_skills, preprocessed_responses

def collate_question_response_fn(batches, pad_val=-1):
    questions = []
    responses = []
    targets = []
    deltas = []

    for batch in batches:
        questions.append(LongTensor(batch["questions"][:-1]))
        responses.append(LongTensor(batch["responses"][:-1]))
        targets.append(FloatTensor(batch["responses"][1:]))
        deltas.append(LongTensor(batch["questions"][1:]))

    """
    pad_sequence를 통해 list of LongTensor가 [B x L (=200)] 의 Tensor로 변환됨
    """
    questions = pad_sequence(
        questions, batch_first=True, padding_value=pad_val
    )
    responses = pad_sequence(
        responses, batch_first=True, padding_value=pad_val
    )
    targets = pad_sequence(
        targets, batch_first=True, padding_value=pad_val
    )
    deltas = pad_sequence(
        deltas, batch_first=True, padding_value=pad_val
    )

    masks = (questions != pad_val) * (deltas != pad_val)

    questions, responses, targets, deltas = \
        questions * masks, responses * masks, targets * masks, \
        deltas * masks


    """
    seq_len=10, pad_val=-1 일때 예제


    questions[10], responses[10], targets[10], deltas[10]
        (tensor([78, 78, 30, 30, 30, 15, 15, 15, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1]),
        tensor([ 1,  1,  1,  1,  1,  0,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1]),
        tensor([ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1., -1., -1., -1., -1., -1., -1.,
                -1., -1., -1., -1., -1., -1.]),
        tensor([78, 30, 30, 30, 15, 15, 15, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1]))

    masks[10]
        tensor([ True,  True,  True,  True,  True,  True,  True,  True, False, False,
                False, False, False, False, False, False, False, False, False, False])

    (questions * masks)[10], (responses*masks)[10], (targets*masks)[10], (deltas*masks)[10]
        (tensor([78, 78, 30, 30, 30, 15, 15, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0]),
        tensor([1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        tensor([1., 1., 1., 1., 0., 1., 1., 1., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
                -0., -0.]),
        tensor([78, 30, 30, 30, 15, 15, 15, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0]))
        
    """

    return questions, responses, targets, deltas, masks

def collate_question_skill_response_fn(batches, pad_val=-1):
    questions = []
    skills = []
    responses = []
    targets = []
    delta_questions = []
    delta_skills = []

    for batch in batches:
        questions.append(LongTensor(batch["questions"][:-1]))
        skills.append(LongTensor(batch["skills"][:-1]))
        responses.append(LongTensor(batch["responses"][:-1]))
        targets.append(FloatTensor(batch["responses"][1:]))
        delta_questions.append(LongTensor(batch["questions"][1:]))
        delta_skills.append(LongTensor(batch["skills"][1:]))

    """
    pad_sequence를 통해 list of LongTensor가 [B x L (=200)] 의 Tensor로 변환됨
    """
    questions = pad_sequence(
        questions, batch_first=True, padding_value=pad_val
    )
    skills = pad_sequence(
        skills, batch_first=True, padding_value=pad_val
    )
    responses = pad_sequence(
        responses, batch_first=True, padding_value=pad_val
    )
    targets = pad_sequence(
        targets, batch_first=True, padding_value=pad_val
    )
    delta_questions = pad_sequence(
        delta_questions, batch_first=True, padding_value=pad_val
    )
    delta_skills = pad_sequence(
        delta_skills, batch_first=True, padding_value=pad_val
    )

    masks = (questions != pad_val) * (delta_questions != pad_val)

    questions, skills, responses, targets, delta_questions, delta_skills = \
        questions * masks, skills * masks, responses * masks, targets * masks, delta_questions * masks, delta_skills * masks



    """
    seq_len=10, pad_val=-1 일때 예제


    questions[10], responses[10], targets[10], deltas[10]
        (tensor([78, 78, 30, 30, 30, 15, 15, 15, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1]),
        tensor([ 1,  1,  1,  1,  1,  0,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1]),
        tensor([ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1., -1., -1., -1., -1., -1., -1.,
                -1., -1., -1., -1., -1., -1.]),
        tensor([78, 30, 30, 30, 15, 15, 15, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1]))

    masks[10]
        tensor([ True,  True,  True,  True,  True,  True,  True,  True, False, False,
                False, False, False, False, False, False, False, False, False, False])

    (questions * masks)[10], (responses*masks)[10], (targets*masks)[10], (deltas*masks)[10]
        (tensor([78, 78, 30, 30, 30, 15, 15, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0]),
        tensor([1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        tensor([1., 1., 1., 1., 0., 1., 1., 1., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
                -0., -0.]),
        tensor([78, 30, 30, 30, 15, 15, 15, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0]))
        
    """

    return questions, skills, responses, targets, delta_questions, delta_skills, masks




def collate_fn(batches):
    questions = []
    skills = []
    responses = []
    lengths = []

    for batch in batches:
        questions.append(LongTensor(batch["questions"]))
        skills.append(LongTensor(batch["skills"]))
        responses.append(LongTensor(batch["responses"]))

    """
    pad_sequence를 통해 list of LongTensor가 [B x L (=200)] 의 Tensor로 변환됨
    """
    questions = pad_sequence(
        questions, batch_first=True, padding_value=0
    )
    skills = pad_sequence(
        skills, batch_first=True, padding_value=0
    )
    responses = pad_sequence(
        responses, batch_first=True, padding_value=-1
    )
    
    feed_dict = {
        "questions": questions,
        "skills": skills,
        "responses": responses
    }
    return feed_dict

def augmented_collate_fn(batches):
    aug_q_seq_1 = []
    aug_q_seq_2 = []
    aug_q_seq = []
    aug_s_seq_1 = []
    aug_s_seq_2 = []
    aug_s_seq = []
    aug_r_seq_1 = []
    aug_r_seq_2 = []
    aug_r_seq = []
    
    for batch in batches:
        aug_q_seq_1.append(LongTensor(batch["questions"][0]))
        aug_q_seq_2.append(LongTensor(batch["questions"][1]))
        aug_q_seq.append(LongTensor(batch["questions"][2]))
        aug_s_seq_1.append(LongTensor(batch["skills"][0]))
        aug_s_seq_2.append(LongTensor(batch["skills"][1]))
        aug_s_seq.append(LongTensor(batch["skills"][2]))
        aug_r_seq_1.append(LongTensor(batch["responses"][0]))
        aug_r_seq_2.append(LongTensor(batch["responses"][1]))
        aug_r_seq.append(LongTensor(batch["responses"][2]))        

    """
    pad_sequence를 통해 list of LongTensor가 [B x L (=200)] 의 Tensor로 변환됨
    """
    aug_q_seq_1 = pad_sequence(
        aug_q_seq_1, batch_first=True, padding_value=0
    )

    aug_q_seq_2 = pad_sequence(
        aug_q_seq_2, batch_first=True, padding_value=0
    )

    aug_q_seq = pad_sequence(
        aug_q_seq, batch_first=True, padding_value=0
    )

    aug_s_seq_1 = pad_sequence(
        aug_s_seq_1, batch_first=True, padding_value=0
    )
    
    aug_s_seq_2 = pad_sequence(
        aug_s_seq_2, batch_first=True, padding_value=0
    )
    
    aug_s_seq = pad_sequence(
        aug_s_seq, batch_first=True, padding_value=0
    )

    aug_r_seq_1 = pad_sequence(
        aug_r_seq_1, batch_first=True, padding_value=-1
    )

    aug_r_seq_2 = pad_sequence(
        aug_r_seq_2, batch_first=True, padding_value=-1
    )
    
    aug_r_seq = pad_sequence(
        aug_r_seq, batch_first=True, padding_value=-1
    )
    feed_dict = {
        "questions": (aug_q_seq_1, aug_q_seq_2, aug_q_seq),
        "skills": (aug_s_seq_1, aug_s_seq_2, aug_s_seq),
        "responses": (aug_r_seq_1, aug_r_seq_2, aug_r_seq)
    }
    return feed_dict