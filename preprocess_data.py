# This code is based on the following repositories:
#  1. https://github.com/theophilee/learner-performance-prediction/blob/master/prepare_data.py
#  2. https://github.com/THUwangcy/HawkesKT/blob/main/data/Preprocess.ipynb

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
import os
import pickle
import time


# Please specify your dataset Path
BASE_PATH = "./dataset"


def prepare_assistments(
    data_name: str, min_user_inter_num: int, remove_nan_skills: bool
):
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
        # df = pd.read_csv(os.path.join(data_path, "skill_builder_data_corrected.csv"), encoding="ISO-8859-1")
        df = df.rename(columns={"problem_id": "item_id"})
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)
    elif data_name == "assistments12":
        # df = pd.read_csv(os.path.join(data_path, "2012-2013-data-with-predictions-4-final.csv"), encoding="ISO-8859-1")
        df = df.rename(columns={"problem_id": "item_id"})
        df["timestamp"] = pd.to_datetime(df["start_time"])
        df["timestamp"] = df["timestamp"] - df["timestamp"].min()
        df["timestamp"] = (
            df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
        )
    elif data_name == "assistments15":
        df = df.rename(columns={"sequence_id": "item_id"})
        df["skill_id"] = df["item_id"]
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)
    elif data_name == "assistments17":
        df = df.rename(
            columns={
                "startTime": "timestamp",
                "studentId": "user_id",
                "problemId": "item_id",
                "skill": "skill_id",
            }
        )
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
    with open(os.path.join(data_path, "skill_id_name"), "wb") as f:
        pickle.dump(dict(zip(df["skill_id"], df["skill_name"])), f)

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
    print("# Interactions: {}".format(len(df)))

    # Get unique skill id from combination of all skill ids
    unique_skill_ids = np.unique(Q_mat, axis=0, return_inverse=True)[1]
    df["skill_id"] = unique_skill_ids[df["item_id"]]

    print("# Preprocessed Skills: {}".format(df["skill_id"].nunique()))
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

    # Save data
    with open(os.path.join(data_path, "question_skill_rel.pkl"), "wb") as f:
        pickle.dump(csr_matrix(Q_mat), f)

    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), csr_matrix(Q_mat))
    df.to_csv(os.path.join(data_path, "preprocessed_df.csv"), sep="\t", index=False)


def prepare_kddcup10(
    data_name: str, min_user_inter_num: int, kc_col_name: str, remove_nan_skills: bool
):
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
    df = pd.read_csv(os.path.join(data_path, "data.txt"), delimiter="\t")
    df = df.rename(
        columns={"Anon Student Id": "user_id", "Correct First Attempt": "correct"}
    )

    # Create item from problem and step
    df["item_id"] = df["Problem Name"] + ":" + df["Step Name"]

    # Add timestamp
    df["timestamp"] = pd.to_datetime(df["First Transaction Time"])
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    df["timestamp"] = (
        df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
    )

    # Remove continuous outcomes
    df = df[df["correct"].isin([0, 1])]
    df["correct"] = df["correct"].astype(np.int32)

    # Filter nan skills
    if remove_nan_skills:
        df = df[~df[kc_col_name].isnull()]
    else:
        df.loc[df[kc_col_name].isnull(), kc_col_name] = "NaN"

    # Drop duplicates
    df.drop_duplicates(subset=["user_id", "item_id", "timestamp"], inplace=True)

    # Filter too short sequences
    df = df.groupby("user_id").filter(lambda x: len(x) >= min_user_inter_num)

    # Extract KCs
    kc_list = []
    for kc_str in df[kc_col_name].unique():
        for kc in kc_str.split("~~"):
            kc_list.append(kc)
    kc_set = set(kc_list)
    kc2idx = {kc: i for i, kc in enumerate(kc_set)}

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]

    print("# Users: {}".format(df["user_id"].nunique()))
    print("# Skills: {}".format(len(kc2idx)))
    print("# Items: {}".format(df["item_id"].nunique()))
    print("# Interactions: {}".format(len(df)))

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(kc_set)))
    for item_id, kc_str in df[["item_id", kc_col_name]].values:
        for kc in kc_str.split("~~"):
            Q_mat[item_id, kc2idx[kc]] = 1

    # Get unique skill id from combination of all skill ids
    unique_skill_ids = np.unique(Q_mat, axis=0, return_inverse=True)[1]
    df["skill_id"] = unique_skill_ids[df["item_id"]]

    print("# Preprocessed Skills: {}".format(df["skill_id"].nunique()))

    # Sort data temporally
    df.sort_values(by="timestamp", inplace=True)

    # Sort data by users, preserving temporal order for each user
    df = pd.concat([u_df for _, u_df in df.groupby("user_id")])
    df.to_csv(os.path.join(data_path, "original_df.csv"), sep="\t", index=False)

    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]
    df.reset_index(inplace=True, drop=True)

    # Save data
    with open(os.path.join(data_path, "question_skill_rel.pkl"), "wb") as f:
        pickle.dump(csr_matrix(Q_mat), f)

    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), csr_matrix(Q_mat))
    df.to_csv(os.path.join(data_path, "preprocessed_df.csv"), sep="\t", index=False)


def prepare_statics():
    data_path = os.path.join(BASE_PATH, "statics")
    df = pd.read_csv(os.path.join(data_path, "interaction_df.csv"), sep="\t")

    df["item_id"] -= 1
    df["skill_id"] -= 1
    print("# Users: {}".format(df["user_id"].nunique()))
    print("# Skills: {}".format(df["skill_id"].nunique()))
    print("# Items: {}".format(df["item_id"].nunique()))
    print("# Interactions: {}".format(len(df)))

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
    print(df["item_id"].min())  # --> 1
    for item_id, skill_id in df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1

    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), csr_matrix(Q_mat))
    df.to_csv(os.path.join(data_path, "preprocessed_df.csv"), sep="\t", index=False)


def prepare_spanish():
    """
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
    print("# Interactions: {}".format(len(df)))

    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]
    df.reset_index(inplace=True, drop=True)

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
    for item_id, skill_id in df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1

    # Sort data by users, preserving temporal order for each user
    df = pd.concat([u_df for _, u_df in df.groupby("user_id")])

    # Save data
    with open(os.path.join(data_path, "question_skill_rel.pkl"), "wb") as f:
        pickle.dump(csr_matrix(Q_mat), f)

    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), csr_matrix(Q_mat))
    df.to_csv(os.path.join(data_path, "preprocessed_df.csv"), sep="\t", index=False)


def prepare_slepemapy(min_user_inter_num):
    """
    This is forked from:
    https://github.com/THUwangcy/HawkesKT/blob/main/data/Preprocess.ipynb
    """

    data_path = os.path.join(BASE_PATH, "slepemapy")
    data_df_cz = pd.read_csv(os.path.join(data_path, "answer.csv"), sep=";")

    # 1. place_answered is NaN
    print("raw data:", len(data_df_cz))
    filter_df_cz = data_df_cz[~data_df_cz["place_answered"].isna()]
    print("drop nan:", len(filter_df_cz))

    # 2. define skill, problem, label
    filter_df_cz.rename(columns={"user": "user_id"}, inplace=True)
    filter_df_cz["correct"] = data_df_cz["place_asked"].astype(float) == data_df_cz[
        "place_answered"
    ].astype(float)
    filter_df_cz["dwell_time"] = filter_df_cz["response_time"].apply(
        lambda t: t / 1000.0
    )
    filter_df_cz["timestamp"] = filter_df_cz["inserted"].apply(
        lambda t: time.mktime(time.strptime(t, "%Y-%m-%d %H:%M:%S"))
    )
    filter_df_cz["skill_id"] = filter_df_cz["place_asked"] - 1
    filter_df_cz["problem_id"] = filter_df_cz["skill_id"] * 2 + filter_df_cz["type"] - 1

    # 3. sequence length is not in a proper range
    user_wise_lst = list()
    for user, user_df in filter_df_cz.groupby("user_id"):
        if len(user_df) >= min_user_inter_num:
            df = user_df.sort_values(by=["timestamp"])  # assure the sequence order
            user_wise_lst.append(df)

    # 4. shuffle
    np.random.shuffle(user_wise_lst)
    user_wise_df_cz = pd.concat(user_wise_lst).reset_index(drop=True)
    user_wise_df_cz = user_wise_df_cz[
        ["user_id", "skill_id", "problem_id", "dwell_time", "timestamp", "correct"]
    ]
    print("drop < {}:".format(min_user_inter_num), len(user_wise_df_cz))
    user_wise_df_cz.head()

    # user re-index
    user_ids = list(user_wise_df_cz["user_id"].unique())
    user_dict = dict(zip(user_ids, range(1, len(user_ids) + 1)))
    user_wise_df_cz["user_id"] = user_wise_df_cz["user_id"].apply(
        lambda x: user_dict[x]
    )
    user_wise_df_cz.head()

    # Adujust dtypes
    user_wise_df_cz = user_wise_df_cz.astype(
        {"correct": np.float64, "dwell_time": np.float64, "timestamp": np.float64}
    )
    user_wise_df_cz.dtypes
    user_wise_df_cz.rename(columns={"problem_id": "item_id"}, inplace=True)

    # item, skill re-index
    user_wise_df_cz["item_id"] = np.unique(
        user_wise_df_cz["item_id"], return_inverse=True
    )[1]
    user_wise_df_cz["skill_id"] = np.unique(
        user_wise_df_cz["skill_id"], return_inverse=True
    )[1]

    # Build Q-matrix
    Q_mat = np.zeros(
        (user_wise_df_cz["item_id"].nunique(), user_wise_df_cz["skill_id"].nunique())
    )
    for item_id, skill_id in user_wise_df_cz[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1

    # Save
    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), csr_matrix(Q_mat))
    user_wise_df_cz.to_csv(
        os.path.join(data_path, "preprocessed_df.csv"), sep="\t", index=False
    )


def prepare_sampled_slepemapy(min_user_inter_num):
    """
    This is forked from:
    https://github.com/THUwangcy/HawkesKT/blob/main/data/Preprocess.ipynb
    """
    data_path = os.path.join(BASE_PATH, "sampled_slepemapy")
    data_df_cz = pd.read_csv(os.path.join(data_path, "answer.csv"), sep=";")

    # 1. place_answered is NaN
    print("raw data:", len(data_df_cz))
    filter_df_cz = data_df_cz[~data_df_cz["place_answered"].isna()]
    print("drop nan:", len(filter_df_cz))

    # 2. define skill, problem, label
    filter_df_cz.rename(columns={"user": "user_id"}, inplace=True)
    filter_df_cz["correct"] = data_df_cz["place_asked"].astype(float) == data_df_cz[
        "place_answered"
    ].astype(float)
    filter_df_cz["dwell_time"] = filter_df_cz["response_time"].apply(
        lambda t: t / 1000.0
    )
    filter_df_cz["timestamp"] = filter_df_cz["inserted"].apply(
        lambda t: time.mktime(time.strptime(t, "%Y-%m-%d %H:%M:%S"))
    )
    filter_df_cz["skill_id"] = filter_df_cz["place_asked"] - 1
    filter_df_cz["problem_id"] = filter_df_cz["skill_id"] * 2 + filter_df_cz["type"] - 1

    # 3. sequence length is not in a proper range
    user_wise_lst = list()
    for user, user_df in filter_df_cz.groupby("user_id"):
        if len(user_df) >= min_user_inter_num:
            df = user_df.sort_values(by=["timestamp"])  # assure the sequence order
            user_wise_lst.append(df)

    # 4. shuffle
    np.random.shuffle(user_wise_lst)
    user_wise_lst = user_wise_lst[:5000]  # sample 5000 students
    user_wise_df_cz = pd.concat(user_wise_lst).reset_index(drop=True)
    user_wise_df_cz = user_wise_df_cz[
        ["user_id", "skill_id", "problem_id", "dwell_time", "timestamp", "correct"]
    ]
    print("drop < {}:".format(min_user_inter_num), len(user_wise_df_cz))
    user_wise_df_cz.head()

    # user re-index
    user_ids = list(user_wise_df_cz["user_id"].unique())
    user_dict = dict(zip(user_ids, range(1, len(user_ids) + 1)))
    user_wise_df_cz["user_id"] = user_wise_df_cz["user_id"].apply(
        lambda x: user_dict[x]
    )
    user_wise_df_cz.head()

    # Adujust dtypes
    user_wise_df_cz = user_wise_df_cz.astype(
        {"correct": np.float64, "dwell_time": np.float64, "timestamp": np.float64}
    )
    user_wise_df_cz.dtypes
    user_wise_df_cz.rename(columns={"problem_id": "item_id"}, inplace=True)

    # item, skill re-index
    user_wise_df_cz["item_id"] = np.unique(
        user_wise_df_cz["item_id"], return_inverse=True
    )[1]
    user_wise_df_cz["skill_id"] = np.unique(
        user_wise_df_cz["skill_id"], return_inverse=True
    )[1]

    # Build Q-matrix
    Q_mat = np.zeros(
        (user_wise_df_cz["item_id"].nunique(), user_wise_df_cz["skill_id"].nunique())
    )
    for item_id, skill_id in user_wise_df_cz[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1

    # Save
    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), csr_matrix(Q_mat))
    user_wise_df_cz.to_csv(
        os.path.join(data_path, "preprocessed_df.csv"), sep="\t", index=False
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Preprocess DKT datasets")
    parser.add_argument("--data_name", type=str, default="assistments09")
    parser.add_argument("--min_user_inter_num", type=int, default=5)
    parser.add_argument("--remove_nan_skills", default=True, action="store_true")
    args = parser.parse_args()

    if args.data_name in [
        "assistments09",
        "assistments12",
        "assistments15",
        "assistments17",
    ]:
        prepare_assistments(
            data_name=args.data_name,
            min_user_inter_num=args.min_user_inter_num,
            remove_nan_skills=args.remove_nan_skills,
        )
    elif args.data_name == "bridge_algebra06":
        prepare_kddcup10(
            data_name="bridge_algebra06",
            min_user_inter_num=args.min_user_inter_num,
            kc_col_name="KC(SubSkills)",
            remove_nan_skills=args.remove_nan_skills,
        )
    elif args.data_name == "algebra05":
        prepare_kddcup10(
            data_name="algebra05",
            min_user_inter_num=args.min_user_inter_num,
            kc_col_name="KC(Default)",
            remove_nan_skills=args.remove_nan_skills,
        )
    elif args.data_name == "spanish":
        prepare_spanish()
    elif args.data_name == "slepemapy":
        prepare_slepemapy(args.min_user_inter_num)
    elif args.data_name == "sampled_slepemapy":
        prepare_sampled_slepemapy(args.min_user_inter_num)
    elif args.data_name == "statics":
        prepare_statics()

