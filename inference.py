from math import e
from random import random
from re import I
import wandb
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
    QuestionSkillDataset,
    MostRecentQuestionSkillDataset,
    MostEarlyQuestionSkillDataset,
    SimCLRDatasetWrapper,
    ConsistencyDataWrapper,
)
from models.sakt import SAKT
from models.akt import AKT
from models.cl4kt_akt import CL4KT
from models.hawkes_kt import HawkesKT
from train import model_train
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
from utils.config import ConfigNode as CN
from utils.file_io import PathManager
from utils.visualizer import draw_heatmap


def inference(config):
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
    elif train_config.sequence_option == "early":
        dataset = MostEarlyQuestionSkillDataset
    elif train_config.sequence_option == "split":  # equivalent to "division"
        """
        CL4KT is not working in this option
        """
        # assert model_name != "cl4kt"
        dataset = QuestionSkillDataset
    else:
        raise NotImplementedError("sequence option is not valid")

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

    for _, (train_ids, test_ids) in enumerate(kfold.split(users)):
        if model_name == "dkt":
            model_config = config.dkt_config
            model = DKT(num_skills, **model_config)
        elif model_name == "dkvmn":
            model_config = config.dkvmn_config
            model = DKVMN(num_skills, seq_len, **model_config)
        elif model_name == "sakt":
            model_config = config.sakt_config
            model = SAKT(num_skills, seq_len, **model_config)
        elif model_name == "akt":
            model_config = config.akt_config
            if data_name in ["statics", "assistments15"]:
                num_questions = 0
            model = AKT(num_skills, num_questions, seq_len, **model_config)
        elif model_name == "hawkes_kt":
            model_config = config.dkt_config
            model = HawkesKT(num_skills, num_questions, seq_len, **model_config)
        elif model_name == "cl4kt":
            model_config = config.cl4kt_config
            model = CL4KT(num_skills, num_questions, seq_len, **model_config)
            mask_prob = model_config.mask_prob
            crop_prob = model_config.crop_prob
            permute_prob = model_config.permute_prob
            replace_prob = model_config.replace_prob
            negative_prob = model_config.negative_prob
        elif model_name == "cl4kt_uni":
            model_config = config.cl4kt_config
            model = CL4KT_Unidirectional(
                num_skills, num_questions, seq_len, **model_config
            )
            mask_prob = model_config.mask_prob
            crop_prob = model_config.crop_prob
            permute_prob = model_config.permute_prob
            replace_prob = model_config.replace_prob
            negative_prob = model_config.negative_prob
        elif model_name == "cl4lstm":
            model_config = config.cl4lstm_config
            model = CL4LSTM(num_skills, num_questions, seq_len, **model_config)
            mask_prob = model_config.mask_prob
            crop_prob = model_config.crop_prob
            permute_prob = model_config.permute_prob
            replace_prob = model_config.replace_prob
            negative_prob = model_config.negative_prob
        elif model_name == "cl4sakt":
            model_config = config.cl4sakt_config
            model = CL4SAKT(num_skills, num_questions, seq_len, **model_config)
            mask_prob = model_config.mask_prob
            crop_prob = model_config.crop_prob
            permute_prob = model_config.permute_prob
            replace_prob = model_config.replace_prob
            negative_prob = model_config.negative_prob
        elif model_name == "dkt_consistency":
            model_config = config.dkt_consistency_config
            model = LSTM_consistency(num_skills, **model_config)
            replace_prob = model_config.replace_prob

        train_users = users[train_ids]
        np.random.shuffle(train_users)
        offset = int(len(train_ids) * 0.9)
        print(offset)
        valid_users = train_users[offset:]
        train_users = train_users[:offset]
        test_users = users[test_ids]

        # train_df = df[df["user_id"].isin(train_users)]
        # valid_df = df[df["user_id"].isin(valid_users)]
        test_df = df[df["user_id"].isin(test_users)]

        # train_dataset = dataset(train_df, seq_len, num_skills, num_questions)
        # valid_dataset = dataset(valid_df, seq_len, num_skills, num_questions)
        test_dataset = dataset(test_df, seq_len, num_skills, num_questions)

        print("train_ids", len(train_users))
        print("valid_ids", len(valid_users))
        print("test_ids", len(test_users))

        if "cl" in model_name:  # contrastive learning
            # train_loader = accelerator.prepare(
            #     DataLoader(
            #         SimCLRDatasetWrapper(
            #             train_dataset, seq_len, mask_prob, crop_prob, permute_prob, replace_prob, negative_prob, eval_mode=False
            #         ),
            #     batch_size=batch_size)
            # )

            # valid_loader = accelerator.prepare(DataLoader(
            #     SimCLRDatasetWrapper(valid_dataset, seq_len, 0, 0, 0, 0, 0, eval_mode=True), batch_size=eval_batch_size,
            # ))

            test_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        test_dataset, seq_len, 0, 0, 0, 0, 0, eval_mode=True
                    ),
                    batch_size=eval_batch_size,
                )
            )
        else:
            # train_loader = accelerator.prepare(DataLoader(
            #     train_dataset, batch_size=batch_size
            # ))

            # valid_loader = accelerator.prepare(DataLoader(
            #     valid_dataset, batch_size=eval_batch_size
            # ))

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

        param_path = os.listdir(os.path.join("saved_model", model_name, data_name))[0]
        print(os.path.join("saved_model", model_name, data_name, param_path))
        checkpoint = torch.load(
            os.path.join("saved_model", model_name, data_name, param_path)
        )

        # best model이 저장되어 있는 last checkpoint를 로드함
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            random_idx = 7
            for batch in test_loader:
                out_dict = model(batch)
                # batch_unif = out_dict["interaction_uniformity"].flatten()
                x = out_dict["x"]
                questions = batch["skills"]
                batch_size = x.shape[0]
                length = x.shape[1]

                total_question_embeds = model.module.question_embed.weight.expand(
                    batch_size, length, -1, -1
                )
                print("total_question_embeds", total_question_embeds.shape)
                print(
                    "x",
                    x.unsqueeze(2)
                    .expand(-1, -1, total_question_embeds.shape[2], -1)
                    .shape,
                )

                retrieved_knowledge = torch.cat(
                    [
                        x.unsqueeze(2).expand(
                            -1, -1, total_question_embeds.shape[2], -1
                        ),
                        total_question_embeds,
                    ],
                    dim=-1,
                )
                print("retrieved knowledge", retrieved_knowledge.shape)

                output = (
                    torch.sigmoid(
                        model.module.out(retrieved_knowledge[random_idx, 1:, :, :])
                    )
                    .squeeze(0)
                    .squeeze(-1)
                )
                print("output", output.shape)
                pred = out_dict["pred"]  # B x L
                true = out_dict["true"]  # B x L
                mask = true > -1
                mask = mask[random_idx, :].squeeze(0)
                pred = pred[random_idx, :].squeeze(0)
                true = true[random_idx, :].squeeze(0)

                pred = pred[mask]
                true = true[mask]
                print("mask", mask.shape)
                output = output[mask, :]  # [L, Q]
                questions = questions[random_idx, :].squeeze(0)[1:]  # [L]
                print("pred", pred.shape)
                print("questions", questions.shape)
                # pred = pred[random_idx,:].squeeze(0)
                # pred = pred[mask]

                draw_heatmap(
                    output.cpu().numpy(),
                    questions.cpu().numpy(),
                    pred.cpu().numpy(),
                    true.cpu().numpy(),
                    f"./heatmap_{random_idx}.png",
                    random_idx,
                )

                # print('questions', questions.shape)

                # print('output', output.shape)
                # responses = batch["responses"]
                # attention_mask = batch["attention_mask"]
                # print('attn_mask', attention_mask.shape)
                # pred = out_dict["pred"]
                # true = out_dict["true"]
                # print('true', true.shape)
                # mask = true > -1
                # pred = pred[mask]
                # true = true[mask]

                # print(len(questions), questions)
                # print(len(pred), pred)
                break
                # total_preds.append(pred)
                # total_trues.append(true)

        # print(model)

        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="dkt",
        help="The name of the model to train. \
            The possible models are in [dkt, dkvmn, sakt, akt]. \
            The default model is dkt.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="assistments09",
        help="The name of the dataset to use in training. \
            The possible datasets are in [assistments09, assistments12, assistments15, assistments17, algebra05, bridge_algebra06, spanish]. \
            The default dataset is assistments2009. Other datasets will be updated.",
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

    if args.model_name in ["dkt", "dkvmn", "akt", "sakt", "dkt_consistency"]:
        cfg.dkt_config.l2 = args.l2
        cfg.dkt_config.dropout = args.dropout
        cfg.dkvmn_config.l2 = args.l2
        cfg.dkvmn_config.dropout = args.dropout
        cfg.akt_config.l2 = args.l2
        cfg.akt_config.dropout = args.dropout
        cfg.sakt_config.l2 = args.l2
        cfg.sakt_config.dropout = args.dropout
        cfg.dkt_consistency_config.l2 = args.l2
        cfg.dkt_consistency_config.dropout = args.dropout

    if args.model_name == "cl4kt":
        cfg.cl4kt_config.reg_cl = args.reg_cl
        cfg.cl4kt_config.mask_prob = args.mask_prob
        cfg.cl4kt_config.crop_prob = args.crop_prob
        cfg.cl4kt_config.permute_prob = args.permute_prob
        cfg.cl4kt_config.replace_prob = args.replace_prob
        cfg.cl4kt_config.negative_prob = args.negative_prob
        cfg.cl4kt_config.dropout = args.dropout
        cfg.cl4kt_config.l2 = args.l2
    if args.model_name == "cl4kt_uni":
        cfg.cl4kt_config.reg_cl = args.reg_cl
        cfg.cl4kt_config.mask_prob = args.mask_prob
        cfg.cl4kt_config.crop_prob = args.crop_prob
        cfg.cl4kt_config.permute_prob = args.permute_prob
        cfg.cl4kt_config.replace_prob = args.replace_prob
        cfg.cl4kt_config.negative_prob = args.negative_prob
        cfg.cl4kt_config.dropout = args.dropout
        cfg.cl4kt_config.l2 = args.l2

    if args.model_name == "cl4lstm":
        cfg.cl4lstm_config.reg_cl = args.reg_cl
        cfg.cl4lstm_config.mask_prob = args.mask_prob
        cfg.cl4lstm_config.crop_prob = args.crop_prob
        cfg.cl4lstm_config.permute_prob = args.permute_prob
        cfg.cl4lstm_config.replace_prob = args.replace_prob
        cfg.cl4lstm_config.negative_prob = args.negative_prob
        cfg.cl4lstm_config.dropout = args.dropout
        cfg.cl4lstm_config.l2 = args.l2
    if args.model_name == "cl4sakt":
        cfg.cl4sakt_config.reg_cl = args.reg_cl
        cfg.cl4sakt_config.mask_prob = args.mask_prob
        cfg.cl4sakt_config.crop_prob = args.crop_prob
        cfg.cl4sakt_config.permute_prob = args.permute_prob
        cfg.cl4sakt_config.replace_prob = args.replace_prob
        cfg.cl4sakt_config.negative_prob = args.negative_prob
        cfg.cl4sakt_config.dropout = args.dropout
        cfg.cl4sakt_config.l2 = args.l2
        # reg_cl 0.1 0.5 1 2 4
        # mask_prob 0.1 0.5 0.9
        # crop_prob 0.1 0.5 0.9
        # permute_prob 0.1 0.5 0.9
        # replace_prob 0.1 0.5 0.9
        # dropout 0.1 0.2
    cfg.freeze()

    print(cfg)
    inference(cfg)
