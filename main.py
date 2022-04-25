import os
import argparse
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import yaml
from data_loaders import (
    MostRecentQuestionSkillDataset,
    MostEarlyQuestionSkillDataset,
    SimCLRDatasetWrapper,
)
from models.akt import AKT
from models.cl4kt import CL4KT
from train import model_train
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
from utils.config import ConfigNode as CN
from utils.file_io import PathManager


def main(config):
    accelerator = Accelerator()
    device = accelerator.device

    model_name = config.model_name
    dataset_path = config.dataset_path
    data_name = config.data_name
    seed = config.seed

    np.random.seed(seed)
    torch.manual_seed(seed)

    df_path = os.path.join(os.path.join(dataset_path, data_name), "preprocessed_df.csv")

    train_config = config.train_config
    checkpoint_dir = config.checkpoint_dir

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    ckpt_path = os.path.join(checkpoint_dir, model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, data_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    batch_size = train_config.batch_size
    eval_batch_size = train_config.eval_batch_size
    learning_rate = train_config.learning_rate
    optimizer = train_config.optimizer
    seq_len = train_config.seq_len

    if train_config.sequence_option == "recent":  # the most recent N interactions
        dataset = MostRecentQuestionSkillDataset
    elif train_config.sequence_option == "early":  # the most early N interactions
        dataset = MostEarlyQuestionSkillDataset
    else:
        raise NotImplementedError("sequence option is not valid")

    test_aucs, test_accs, test_rmses = [], [], []

    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

    df = pd.read_csv(df_path, sep="\t")

    print("skill_min", df["skill_id"].min())
    users = df["user_id"].unique()
    df["skill_id"] += 1  # zero for padding
    df["item_id"] += 1  # zero for padding
    num_skills = df["skill_id"].max() + 1
    num_questions = df["item_id"].max() + 1
    np.random.shuffle(users)

    print("MODEL", model_name)
    print(dataset)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(users)):
        if model_name == "akt":
            model_config = config.akt_config
            if data_name in ["statics", "assistments15"]:
                num_questions = 0
            model = AKT(num_skills, num_questions, seq_len, **model_config)
        elif model_name == "cl4kt":
            model_config = config.cl4kt_config
            model = CL4KT(num_skills, num_questions, seq_len, **model_config)
            mask_prob = model_config.mask_prob
            crop_prob = model_config.crop_prob
            permute_prob = model_config.permute_prob
            replace_prob = model_config.replace_prob
            negative_prob = model_config.negative_prob

        train_users = users[train_ids]
        np.random.shuffle(train_users)
        offset = int(len(train_ids) * 0.9)

        valid_users = train_users[offset:]
        train_users = train_users[:offset]
        test_users = users[test_ids]

        train_df = df[df["user_id"].isin(train_users)]
        valid_df = df[df["user_id"].isin(valid_users)]
        test_df = df[df["user_id"].isin(test_users)]

        train_dataset = dataset(train_df, seq_len, num_skills, num_questions)
        valid_dataset = dataset(valid_df, seq_len, num_skills, num_questions)
        test_dataset = dataset(test_df, seq_len, num_skills, num_questions)

        print("train_ids", len(train_users))
        print("valid_ids", len(valid_users))
        print("test_ids", len(test_users))

        if "cl" in model_name:  # contrastive learning
            train_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        train_dataset,
                        seq_len,
                        mask_prob,
                        crop_prob,
                        permute_prob,
                        replace_prob,
                        negative_prob,
                        eval_mode=False,
                    ),
                    batch_size=batch_size,
                )
            )

            valid_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        valid_dataset, seq_len, 0, 0, 0, 0, 0, eval_mode=True
                    ),
                    batch_size=eval_batch_size,
                )
            )

            test_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        test_dataset, seq_len, 0, 0, 0, 0, 0, eval_mode=True
                    ),
                    batch_size=eval_batch_size,
                )
            )
        else:
            train_loader = accelerator.prepare(
                DataLoader(train_dataset, batch_size=batch_size)
            )

            valid_loader = accelerator.prepare(
                DataLoader(valid_dataset, batch_size=eval_batch_size)
            )

            test_loader = accelerator.prepare(
                DataLoader(test_dataset, batch_size=eval_batch_size)
            )

        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate, weight_decay=model_config.l2)

        model, opt = accelerator.prepare(model, opt)

        test_auc, test_acc, test_rmse = model_train(
            fold,
            model,
            accelerator,
            opt,
            train_loader,
            valid_loader,
            test_loader,
            config,
            n_gpu,
        )

        test_aucs.append(test_auc)
        test_accs.append(test_acc)
        test_rmses.append(test_rmse)

    test_auc = np.mean(test_aucs)
    test_auc_std = np.std(test_aucs)
    test_acc = np.mean(test_accs)
    test_acc_std = np.std(test_accs)
    test_rmse = np.mean(test_rmses)
    test_rmse_std = np.std(test_rmses)

    now = (datetime.now() + timedelta(hours=9)).strftime("%Y%m%d-%H%M%S")  # KST time

    log_out_path = os.path.join(
        os.path.join("logs", "5-fold-cv", "{}".format(data_name))
    )
    os.makedirs(log_out_path, exist_ok=True)
    with open(os.path.join(log_out_path, "{}-{}".format(model_name, now)), "w") as f:
        f.write("AUC\tACC\tRMSE\n")
        f.write("{:.5f}\t{:.5f}\t{:.5f}".format(test_auc, test_acc, test_rmse))

    print("\n5-fold CV Result")
    print("AUC\tACC\tRMSE")
    print("{:.5f}\t{:.5f}\t{:.5f}".format(test_auc, test_acc, test_rmse))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="cl4kt",
        help="The name of the model to train. \
            The possible models are in [akt, cl4kt]. \
            The default model is cl4kt.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="algebra05",
        help="The name of the dataset to use in training.",
    )
    parser.add_argument(
        "--reg_cl",
        type=float,
        default=0.1,
        help="regularization parameter contrastive learning loss",
    )
    parser.add_argument("--mask_prob", type=float, default=0.2, help="mask probability")
    parser.add_argument("--crop_prob", type=float, default=0.3, help="crop probability")
    parser.add_argument(
        "--permute_prob", type=float, default=0.3, help="permute probability"
    )
    parser.add_argument(
        "--replace_prob", type=float, default=0.3, help="replace probability"
    )
    parser.add_argument(
        "--negative_prob",
        type=float,
        default=1.0,
        help="reverse responses probability for hard negative pairs",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="dropout probability"
    )
    parser.add_argument(
        "--batch_size", type=float, default=512, help="train batch size"
    )
    parser.add_argument("--l2", type=float, default=0.0, help="l2 regularization param")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    args = parser.parse_args()

    base_cfg_file = PathManager.open("configs/example.yaml", "r")
    base_cfg = yaml.safe_load(base_cfg_file)
    cfg = CN(base_cfg)
    cfg.set_new_allowed(True)
    cfg.model_name = args.model_name
    cfg.data_name = args.data_name
    cfg.train_config.batch_size = int(args.batch_size)
    cfg.train_config.learning_rate = args.lr
    cfg.train_config.optimizer = args.optimizer

    if args.model_name == "cl4kt":
        cfg.cl4kt_config.reg_cl = args.reg_cl
        cfg.cl4kt_config.mask_prob = args.mask_prob
        cfg.cl4kt_config.crop_prob = args.crop_prob
        cfg.cl4kt_config.permute_prob = args.permute_prob
        cfg.cl4kt_config.replace_prob = args.replace_prob
        cfg.cl4kt_config.negative_prob = args.negative_prob
        cfg.cl4kt_config.dropout = args.dropout
        cfg.cl4kt_config.l2 = args.l2
    else:  # akt
        cfg.akt_config.l2 = args.l2
        cfg.akt_config.dropout = args.dropout

    cfg.freeze()

    print(cfg)
    main(cfg)
