from pickle import TUPLE2
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
import random
from utils import preprocess_qr, preprocess_qsr, augment_kt_seqs

# https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/
class SimCLRDatasetWrapper(Dataset):
    def __init__(self, ds: Dataset, q_mask_prob: float, s_mask_prob: float, r_mask_prob: float, eval_mode=False):
        super().__init__()
        self.ds = ds
        self.q_mask_prob = q_mask_prob
        self.s_mask_prob = s_mask_prob
        self.r_mask_prob = r_mask_prob
        self.eval_mode = eval_mode

        self.num_users = self.ds.num_users
        self.num_questions = self.ds.num_questions
        self.num_skills = self.ds.num_skills
        """
        +1 해줘야 하나? embedding에서 padding_idx=0을 할 때,
        원래 데이터 상의 idx=0 인 애들을 어떻게 처리할지... +1로 불러올지에 따라 다를 듯?
        """
        self.q_mask_id = self.num_questions 
        self.s_mask_id = self.num_skills
        self.r_mask_id = 2

    def __len__(self):
        return len(self.ds)

    def __getitem_internal__(self, index):
        """
        이 부분이 메인임... corruption, negative sampling을 어떻게 수행할지...
        뭔가 attrition bias등.. 교육학 이론을 고려해서 augmentation할 순 없을까?
        """
        original_data = self.ds[index]
        q_seq = original_data["questions"]
        s_seq = original_data["skills"]
        r_seq = original_data["responses"]

        if self.eval_mode:
            return {
                "questions": q_seq,
                "skills": s_seq,
                "responses": r_seq
            }

        else:
            # do augmentation
            # masking questions
            # corrupt responses? (negative pair?)
            # how to define negative questions and responses?
            # SimCLR은 explicit한 negative sampling은 없음
            # batch 내에 동일 샘플에서 기인하지 않은 augmented sample을 negative로 취급할 뿐
            t1 = augment_kt_seqs(q_seq, s_seq, r_seq, self.q_mask_prob, self.s_mask_prob, self.r_mask_prob,
                            self.q_mask_id, self.s_mask_id, self.r_mask_id, seed=index)
            
            t2 = augment_kt_seqs(q_seq, s_seq, r_seq, self.q_mask_prob, self.s_mask_prob, self.r_mask_prob,
                            self.q_mask_id, self.s_mask_id, self.r_mask_id, seed=index+1)

            aug_q_seq_1, aug_s_seq_1, aug_r_seq_1 = t1
            aug_q_seq_2, aug_s_seq_2, aug_r_seq_2 = t2


            return {
                "questions": (aug_q_seq_1, aug_q_seq_2, q_seq),
                "skills": (aug_s_seq_1, aug_s_seq_2, s_seq),
                "responses": (aug_r_seq_1, aug_r_seq_2, r_seq)
            }
        

    def __getitem__(self, index):
        return self.__getitem_internal__(index)



class MostRecentQuestionSkillDataset(Dataset):
    def __init__(self, df_path, seq_len):
        self.df_path = df_path
        self.seq_len = seq_len
        self.df = pd.read_csv(self.df_path, sep="\t")
        
        self.unique_users = self.df["user_id"].unique()
        self.unique_questions = self.df["item_id"].unique()
        self.unique_skills = self.df["skill_id"].unique()

        self.num_users = self.df["user_id"].max()+1
        self.num_questions = self.df["item_id"].max()+1
        self.num_skills = self.df["skill_id"].max()+1

        self.questions = [u_df["item_id"].values[-self.seq_len:] for _, u_df in self.df.groupby("user_id")]
        self.skills = [u_df["skill_id"].values[-self.seq_len:] for _, u_df in self.df.groupby("user_id")]
        self.responses = [u_df["correct"].values[-self.seq_len:] for _, u_df in self.df.groupby("user_id")]
        self.lengths = [len(u_df["skill_id"].values) for _, u_df in self.df.groupby("user_id")]

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt
        
        self.questions, self.skills, self.responses = preprocess_qsr(self.questions, self.skills, self.responses, self.seq_len)
        self.len = len(self.questions)

        print("Num users: {}".format(self.num_users))
        print("Num questions: {}".format(self.num_questions))
        print("Num skills: {}".format(self.num_skills))
        print("Num interactions: {}".format(self.num_interactions))
        print("Avg length of interactions: {}".format(np.mean(self.lengths)))

    def __getitem__(self, index):
        return {"questions": self.questions[index],
                "skills": self.skills[index],
                "responses": self.responses[index]
               }
    
    def __len__(self):
        return self.len

    def __str__(self):
        return "Descriptive Statistics\nㄴdf_path: {}\n\tㄴseq_len: {}\n\tㄴnum_users: {}\n\tㄴnumser_questions: {}\n\tㄴnum_skills: {}\n\tㄴnum_instances: {}\n\tㄴnum_interactions: {}".format(
            self.df_path, self.seq_len, self.num_users, self.num_questions, self.num_skills, self.len, self.num_interactions
        )


class SkillDataset(Dataset):
    def __init__(self, df_path, seq_len):
        """
        df는 이미 정렬된 데이터임 -> user_id로 groupby만 하면 됨
        :param df_path: (str) (e.g., "dataset/assistments09/preprocessed_train_df.csv")
            columns = ["user_id","item_id","skill_id","timestamp","correct"]
        :
        """
        self.df_path = df_path
        self.seq_len = seq_len
        self.df = pd.read_csv(self.df_path, sep="\t")
        
        self.unique_users = self.df["user_id"].unique()
        self.unique_questions = self.df["skill_id"].unique()

        self.num_users = self.df["user_id"].max()+1
        self.num_questions = self.df["skill_id"].max()+1
        """
        이미 zero-based indexing으로 맵핑된 값이 들어옴
        """

        self.questions = [u_df["skill_id"].values for _, u_df in self.df.groupby("user_id")]
        self.responses = [u_df["correct"].values for _, u_df in self.df.groupby("user_id")]
        self.lengths = [len(u_df["skill_id"].values) for _, u_df in self.df.groupby("user_id")]

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt

        self.questions, self.responses = preprocess_qr(self.questions, self.responses, self.seq_len)
        self.len = len(self.questions)


        print("Num users: {}".format(self.num_users))
        print("Num skills: {}".format(self.num_questions))
        print("Num interactions: {}".format(self.num_interactions))
        print("Avg length of interactions: {}".format(np.mean(self.lengths)))


    def __getitem__(self, index):
        return {"questions": self.questions[index],
                "responses": self.responses[index]}

    def __len__(self):
        return self.len

    def __str__(self):
        return "Descriptive Statistics\nㄴdf_path: {}\n\tㄴseq_len: {}\n\tㄴnum_users: {}\n\tㄴnumser_questions: {}\n\tㄴnum_instances: {}\n\tㄴnum_interactions: {}".format(
            self.df_path, self.seq_len, self.num_users, self.num_questions, self.len, self.num_interactions
        )

class QuestionDataset(Dataset):
    def __init__(self, df_path, seq_len):
        """
        df는 이미 정렬된 데이터임 -> user_id로 groupby만 하면 됨
        :param df_path: (str) (e.g., "dataset/assistments09/preprocessed_train_df.csv")
            columns = ["user_id","item_id","skill_id","timestamp","correct"]
        :
        """
        self.df_path = df_path
        self.seq_len = seq_len
        self.df = pd.read_csv(self.df_path, sep="\t")
        
        self.unique_users = self.df["user_id"].unique()
        self.unique_questions = self.df["item_id"].unique()

        self.num_users = self.df["user_id"].max()+1
        self.num_questions = self.df["item_id"].max()+1

        self.questions = [u_df["item_id"].values for _, u_df in self.df.groupby("user_id")]
        self.responses = [u_df["correct"].values for _, u_df in self.df.groupby("user_id")]
        self.lengths = [len(u_df["skill_id"].values) for _, u_df in self.df.groupby("user_id")]

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt
        
        self.questions, self.responses = preprocess_qr(self.questions, self.responses, self.seq_len)
        self.len = len(self.questions)

        print("Num users: {}".format(self.num_users))
        print("Num questions: {}".format(self.num_questions))
        print("Num interactions: {}".format(self.num_interactions))
        print("Avg length of interactions: {}".format(np.mean(self.lengths)))

    def __getitem__(self, index):
        return {"questions": self.questions[index],
                "responses": self.responses[index]}
    
    def __len__(self):
        return self.len

    def __str__(self):
        return "Descriptive Statistics\nㄴdf_path: {}\n\tㄴseq_len: {}\n\tㄴnum_users: {}\n\tㄴnumser_questions: {}\n\tㄴnum_instances: {}\n\tㄴnum_interactions: {}".format(
            self.df_path, self.seq_len, self.num_users, self.num_questions, self.len, self.num_interactions
        )


class QuestionSkillDataset(Dataset):
    def __init__(self, df_path, seq_len):
        """
        df는 이미 정렬된 데이터임 -> user_id로 groupby만 하면 됨
        :param df_path: (str) (e.g., "dataset/assistments09/preprocessed_train_df.csv")
            columns = ["user_id","item_id","skill_id","timestamp","correct"]
        :
        """
        self.df_path = df_path
        self.seq_len = seq_len
        self.df = pd.read_csv(self.df_path, sep="\t")
        
        self.unique_users = self.df["user_id"].unique()
        self.unique_questions = self.df["item_id"].unique()
        self.unique_skills = self.df["skill_id"].unique()

        self.num_users = self.df["user_id"].max()+1
        self.num_questions = self.df["item_id"].max()+1
        self.num_skills = self.df["skill_id"].max()+1

        self.questions = [u_df["item_id"].values for _, u_df in self.df.groupby("user_id")]
        self.skills = [u_df["skill_id"].values for _, u_df in self.df.groupby("user_id")]
        self.responses = [u_df["correct"].values for _, u_df in self.df.groupby("user_id")]
        self.lengths = [len(u_df["skill_id"].values) for _, u_df in self.df.groupby("user_id")]

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt
        
        self.questions, self.skills, self.responses = preprocess_qsr(self.questions, self.skills, self.responses, self.seq_len)
        self.len = len(self.questions)

        print("Num users: {}".format(self.num_users))
        print("Num questions: {}".format(self.num_questions))
        print("Num skills: {}".format(self.num_skills))
        print("Num interactions: {}".format(self.num_interactions))
        print("Avg length of interactions: {}".format(np.mean(self.lengths)))

    def __getitem__(self, index):
        return {"questions": self.questions[index],
                "skills": self.skills[index],
                "responses": self.responses[index]
               }
    
    def __len__(self):
        return self.len

    def __str__(self):
        return "Descriptive Statistics\nㄴdf_path: {}\n\tㄴseq_len: {}\n\tㄴnum_users: {}\n\tㄴnumser_questions: {}\n\tㄴnum_skills: {}\n\tㄴnum_instances: {}\n\tㄴnum_interactions: {}".format(
            self.df_path, self.seq_len, self.num_users, self.num_questions, self.num_skills, self.len, self.num_interactions
        )



if __name__=="__main__":
    df_path = os.path.join(os.path.join("dataset", "algebra05"), "preprocessed_df.csv")
    seq_len = 100
    dataset = QuestionDataset(df_path, seq_len)
    print(dataset)