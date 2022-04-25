"""
jedm prepare_dataset.py 를 참고해서 public benchmark dataset을 hcnoh dataset에서 로딩할 수 있는 형태의
CSV 파일로 덤프

TODO: 5 or 10-fold CV 할 수 있도록 train_test_split 수정
"""

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, data
import os
import pickle

BASE_PATH = "dataset"


def prepare_assistments(data_name: str, min_user_inter_num: int, remove_nan_skills: bool, train_split: float=0.8):
    """
    Preprocess ASSISTments dataset

        :param data_name: (str) "assistments09", "assistments12", "assisments15", and "assistments17"
        :param min_user_inter_num: (int) Users whose number of interactions is less than min_user_inter_num will be removed
        :param remove_nan_skills: (bool) if True, remove interactions with no skill tage
        :param train_split: (float) proportion of data to use for training
        
        :output df: (pd.DataFrame) preprocssed ASSISTments dataset with user_id, item_id, timestamp, correct and unique skill features
        :output question_skill_rel: (csr_matrix) corresponding question-skill relationship matrix
    """
    data_path = os.path.join(BASE_PATH, data_name)
    df = pd.read_csv(os.path.join(data_path, "data.csv"), encoding="ISO-8859-1")

    # Only 2012 and 2017 versions have timestamps
    if data_name == "assistments09":
        df = df.rename(columns={"problem_id": "item_id"})
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)
    elif data_name == "assistments12":
        df = df.rename(columns={"problem_id": "item_id"})
        df["timestamp"] = pd.to_datetime(df["start_time"])
        df["timestamp"] = df["timestamp"] - df["timestamp"].min()
        df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
    elif data_name == "assistments15":
        df = df.rename(columns={"sequence_id": "item_id"})
        df["skill_id"] = df["item_id"]
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)
    elif data_name == "assistments17":
        df = df.rename(columns={"startTime": "timestamp",
                                "studentId": "user_id",
                                "problemId": "item_id",
                                "skill": "skill_id"})
        df["timestamp"] = df["timestamp"] - df["timestamp"].min()

    # Remove continuous outcomes
    df = df[df["correct"].isin([0, 1])]
    df["correct"] = df["correct"].astype(np.int32)

    # Filter nan skills
    if remove_nan_skills:
        df = df[~df["skill_id"].isnull()]
    else:
        df.loc[df["skill_id"].isnull(), "skill_id"] = -1

    # Filter too short sequences
    df = df.groupby("user_id").filter(lambda x: len(x) >= min_user_inter_num)

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]
    df["skill_id"] = np.unique(df["skill_id"].astype(str), return_inverse=True)[1]

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
    for item_id, skill_id in df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1

    # Remove row duplicates due to multiple skills for one item
    if data_name == "assistments09":
        df = df.drop_duplicates("order_id")
    elif data_name == "assistments17":
        df = df.drop_duplicates(["user_id", "timestamp"])

    print("# Users: {}".format(df["user_id"].nunique()))
    print("# Skills: {}".format(df["skill_id"].nunique()))
    print("# Items: {}".format(df["item_id"].nunique()))
    # Get unique skill id from combination of all skill ids
    unique_skill_ids = np.unique(Q_mat, axis=0, return_inverse=True)[1]
    df["skill_id"] = unique_skill_ids[df["item_id"]]

    # Sort data temporally
    if data_name in ["assistments12", "assistments17"]:
        df.sort_values(by="timestamp", inplace=True)
    elif data_name == "assistments09":
        df.sort_values(by="order_id", inplace=True)
    elif data_name == "assistments15":
        df.sort_values(by="log_id", inplace=True)

    # Sort data by users, preserving temporal order for each user
    df = pd.concat([u_df for _, u_df in df.groupby("user_id")])
    df.to_csv(os.path.join(data_path, "original_df.csv"), sep="\t", index=False)

    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]
    df.reset_index(inplace=True, drop=True)

    # Text files for BKT implementation (https://github.com/robert-lindsey/WCRP/)
    bkt_dataset = df[["user_id", "item_id", "correct"]]
    bkt_skills = unique_skill_ids
    bkt_split = np.random.randint(low=0, high=5, size=df["user_id"].nunique()).reshape(1, -1)

    # Train-test split
    users = df["user_id"].unique()
    np.random.shuffle(users)
    split = int(train_split * len(users))
    train_df = df[df["user_id"].isin(users[:split])]
    test_df = df[df["user_id"].isin(users[split:])]

    # Save data

    with open(os.path.join(data_path, "question_skill_rel.pkl"), "wb") as f:
        pickle.dump(csr_matrix(Q_mat), f)

    with open(os.path.join(data_path, "bkt_data.pkl"), "wb") as f:
        pickle.dump((bkt_dataset, bkt_skills, bkt_split), f)

    df.to_csv(os.path.join(data_path, "preprocessed_df.csv"), sep="\t", index=False)
    train_df.to_csv(os.path.join(data_path, "preprocessed_train_df.csv"), sep="\t", index=False)
    test_df.to_csv(os.path.join(data_path, "preprocessed_test_df.csv"), sep="\t", index=False)




def prepare_kddcup10(data_name: str, min_user_inter_num: int, kc_col_name: str, remove_nan_skills: bool, train_split: float=0.8):
    """
    Preprocess KDD Cup 2010 dataset

        :param data_name: (str) "bridge_algebra06" or "algebra05"
        :param min_user_inter_num: (int) Users whose number of interactions is less than min_user_inter_num will be removed
        :param kc_col_name: (str) Skills id column
        :param remove_nan_skills: (bool) if True, remove interactions with no skill tage
        :param train_split: (float) proportion of data to use for training

        :output df: (pd.DataFrame) preprocssed ASSISTments dataset with user_id, item_id, timestamp, correct and unique skill features
        :output question_skill_rel: (csr_matrix) corresponding question-skill relationship matrix
    """
    data_path = os.path.join(BASE_PATH, data_name)
    df = pd.read_csv(os.path.join(data_path, "data.txt"), delimiter='\t')
    df = df.rename(columns={'Anon Student Id': 'user_id',
                            'Correct First Attempt': 'correct'})

    # Create item from problem and step
    df["item_id"] = df["Problem Name"] + ":" + df["Step Name"]

    # Add timestamp
    df["timestamp"] = pd.to_datetime(df["First Transaction Time"])
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)

    # Remove continuous outcomes
    df = df[df["correct"].isin([0, 1])]
    df['correct'] = df['correct'].astype(np.int32)

    # Filter nan skills
    if remove_nan_skills:
        df = df[~df[kc_col_name].isnull()]
    else:
        df.loc[df[kc_col_name].isnull(), kc_col_name] = 'NaN'

    # Drop duplicates
    df.drop_duplicates(subset=["user_id", "item_id", "timestamp"], inplace=True)

    # Filter too short sequences
    df = df.groupby("user_id").filter(lambda x: len(x) >= min_user_inter_num)

    # Extract KCs
    kc_list = []
    for kc_str in df[kc_col_name].unique():
        for kc in kc_str.split('~~'):
            kc_list.append(kc)
    kc_set = set(kc_list)
    kc2idx = {kc: i for i, kc in enumerate(kc_set)}

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]


    print("# Users: {}".format(df["user_id"].nunique()))
    print("# Skills: {}".format(df[kc_col_name].nunique()))
    print("# Items: {}".format(df["item_id"].nunique()))


    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(kc_set)))
    for item_id, kc_str in df[["item_id", kc_col_name]].values:
        for kc in kc_str.split('~~'):
            Q_mat[item_id, kc2idx[kc]] = 1

    # Get unique skill id from combination of all skill ids
    unique_skill_ids = np.unique(Q_mat, axis=0, return_inverse=True)[1]
    df["skill_id"] = unique_skill_ids[df["item_id"]]


    # Sort data temporally
    df.sort_values(by="timestamp", inplace=True)

    # Sort data by users, preserving temporal order for each user
    df = pd.concat([u_df for _, u_df in df.groupby("user_id")])
    df.to_csv(os.path.join(data_path, "original_df.csv"), sep="\t", index=False)

    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]
    df.reset_index(inplace=True, drop=True)


    # Text files for BKT implementation (https://github.com/robert-lindsey/WCRP/)
    bkt_dataset = df[["user_id", "item_id", "correct"]]
    bkt_skills = unique_skill_ids
    bkt_split = np.random.randint(low=0, high=5, size=df["user_id"].nunique()).reshape(1, -1)

    # Train-test split
    users = df["user_id"].unique()
    np.random.shuffle(users)
    split = int(train_split * len(users))
    train_df = df[df["user_id"].isin(users[:split])]
    test_df = df[df["user_id"].isin(users[split:])]

    # Save data
    with open(os.path.join(data_path, "question_skill_rel.pkl"), "wb") as f:
        pickle.dump(csr_matrix(Q_mat), f)

    with open(os.path.join(data_path, "bkt_data.pkl"), "wb") as f:
        pickle.dump((bkt_dataset, bkt_skills, bkt_split), f)


    df.to_csv(os.path.join(data_path, "preprocessed_df.csv"), sep="\t", index=False)
    train_df.to_csv(os.path.join(data_path, "preprocessed_train_df.csv"), sep="\t", index=False)
    test_df.to_csv(os.path.join(data_path, "preprocessed_test_df.csv"), sep="\t", index=False)



def prepare_spanish(train_split=0.8):
    """
    TODO: Spanish 데이터는 따로 필터링을 안하나 확인해보기, BKT preprocessing도 따로 안하네?

    Preprocess Spanish dataset.
    :param train_split: (float) proportion of data to use for training
    :output df: (pd.DataFrame) preprocessed dataset with user_id, item_id, timestamp, correct and unique skill features
    :output question_skill_rel: (csr_matrix) question-skill relationship sparse matrix
    """
    data_path = os.path.join(BASE_PATH, "spanish")

    data = np.loadtxt(os.path.join(data_path, "spanish_dataset.txt"), dtype=int)
    df = pd.DataFrame(data=data, columns=("user_id", "item_id", "correct"))

    skills = np.loadtxt(os.path.join(data_path, "spanish_expert_labels.txt"))
    df["skill_id"] = skills[df["item_id"]].astype(np.int64)

    df["timestamp"] = np.zeros(len(df), np.int64)

    print("# Users: {}".format(df["user_id"].nunique()))
    print("# Skills: {}".format(df["skill_id"].nunique()))
    print("# Items: {}".format(df["item_id"].nunique()))

    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]
    df.reset_index(inplace=True, drop=True)


    # Build Q-matrix
    Q_mat = np.zeros((df["item_id"].nunique(), df["skill_id"].nunique()))
    for item_id, skill_id in df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1

    # Sort data by users, preserving temporal order for each user
    df = pd.concat([u_df for _, u_df in df.groupby("user_id")])

    """
    Train-test split은 pytorch data utils 활용해서 처리
    """
    # users = df["user_id"].unique()
    # np.random.shuffle(users)
    # split = int(train_split * len(users))
    # train_df = df[df["user_id"].isin(users[:split])]
    # test_df = df[df["user_id"].isin(users[split:])]


    # Save data
    with open(os.path.join(data_path, "question_skill_rel.pkl"), "wb") as f:
        pickle.dump(csr_matrix(Q_mat), f)

    df.to_csv(os.path.join(data_path, "preprocessed_df.csv"), sep="\t", index=False)
    # train_df.to_csv(os.path.join(data_path, "preprocessed_train_df.csv"), sep="\t", index=False)
    # test_df.to_csv(os.path.join(data_path, "preprocessed_test_df.csv"), sep="\t", index=False)



if __name__=="__main__":
    parser = ArgumentParser(description="Preprocess DKT datasets")
    parser.add_argument('--dataset', type=str, default='assistments09')
    parser.add_argument('--min_user_inter_num', type=int, default=10)
    parser.add_argument('--remove_nan_skills', default=True, action='store_true')
    args = parser.parse_args()

    if args.dataset in ["assistments09", "assistments12", "assistments15", "assistments17"]:
        prepare_assistments(
            data_name=args.dataset,
            min_user_inter_num=args.min_user_inter_num,
            remove_nan_skills=args.remove_nan_skills)
    elif args.dataset == "bridge_algebra06":
        prepare_kddcup10(
            data_name="bridge_algebra06",
            min_user_inter_num=args.min_user_inter_num,
            kc_col_name="KC(SubSkills)",
            remove_nan_skills=args.remove_nan_skills)
    elif args.dataset == "algebra05":
        prepare_kddcup10(
            data_name="algebra05",
            min_user_inter_num=args.min_user_inter_num,
            kc_col_name="KC(Default)",
            remove_nan_skills=args.remove_nan_skills)
    elif args.dataset == "spanish":
        prepare_spanish()

