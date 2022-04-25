"""
BKT, PFA (from JEDM)
DKT, DKVMN, SAKT (from hcnoh)
AKT, Hawkes KT (from ddd)


"""
import os
import argparse
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import SGD, Adam
import yaml
from data_loaders import QuestionSkillDataset, MostRecentQuestionSkillDataset, SimCLRDatasetWrapper
from models.dkt import DKT
from models.dkvmn import DKVMN
from models.sakt import SAKT
from models.akt import AKT
from models.cl4kt import CL4KT
from utils import collate_fn, augmented_collate_fn
from train import model_train
from sklearn.model_selection import KFold
from datetime import datetime, timedelta


def main(config):


    accelerator = Accelerator()
    device = accelerator.device

    model_name = config["model_name"]
    dataset_path = config["dataset_path"]
    data_name = config["data_name"]
    seed = config["seed"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    df_path = os.path.join(os.path.join(dataset_path, data_name), "interactions.csv")


    train_config = config["train_config"]
    checkpoint_dir = config["checkpoint_dir"]

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    ckpt_path = os.path.join(checkpoint_dir, model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, data_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    batch_size = train_config["batch_size"]
    eval_batch_size = train_config["eval_batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    learning_rate = train_config["learning_rate"]
    optimizer = train_config["optimizer"]  # can be [sgd, adam]
    loss = train_config["loss"]
    seq_len = train_config["seq_len"]
    if train_config["sequence_option"] == "recent": # the most recent N interactions
        dataset = MostRecentQuestionSkillDataset
    elif train_config["sequence_option"] == "split": # equivalent to "division"
        """
        CL4KT is not working in this option
        """
        assert model_name != "cl4kt"
        
        dataset = QuestionSkillDataset

    if model_name == "dkt":
        dataset = dataset(df_path, seq_len)
        model_config = config["dkt_config"]
        model = DKT(dataset.num_skills, **model_config)
    elif model_name == "dkvmn":
        dataset = dataset(df_path, seq_len)
        model_config = config["dkvmn_config"]
        model = DKVMN(dataset.num_skills, **model_config)
    elif model_name == "sakt":
        dataset = dataset(df_path, seq_len)
        model_config = config["sakt_config"]
        model = SAKT(dataset.num_skills, seq_len, **model_config)
    elif model_name == "akt":
        dataset = dataset(df_path, seq_len)
        model_config = config["akt_config"]
        model = AKT(dataset.num_skills, dataset.num_questions, seq_len, **model_config)
    elif model_name == "cl4kt":
        dataset = dataset(df_path, seq_len)
        model_config = config["cl4kt_config"]
        model = CL4KT(dataset.num_skills, dataset.num_questions, seq_len, **model_config)
        q_mask_prob = model_config["q_mask_prob"]
        s_mask_prob = model_config["s_mask_prob"]
        r_mask_prob = model_config["r_mask_prob"]
    print("MODEL", model_name)
    print(dataset)


    """
    아래의 random split을 K-fold CV로 변경하기
    torch.utils.data.Subset() 사용해서 specified indices를 통해 임의의 train, test 받아올 수 있음
    """

    test_aucs, test_accs, test_rmses = [], [], []

    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        offset = int(len(train_ids) * 0.8)
        valid_ids = train_ids[offset:]
        train_ids = train_ids[:offset]

        print('train_ids', len(train_ids))
        print('valid_ids', len(valid_ids))
        print('test_ids', len(test_ids))
    
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)
        test_subsampler = SubsetRandomSampler(test_ids)
        
        if model_name == "cl4kt":
            print('here')
            train_loader = accelerator.prepare(DataLoader(
                SimCLRDatasetWrapper(dataset, q_mask_prob, s_mask_prob, r_mask_prob, eval_mode=False), batch_size=batch_size,
                collate_fn=augmented_collate_fn, sampler=train_subsampler
            ))

            valid_loader = accelerator.prepare(DataLoader(
                SimCLRDatasetWrapper(dataset, 0, 0, 0, eval_mode=True), batch_size=eval_batch_size,
                collate_fn=collate_fn, sampler=valid_subsampler
            ))

            test_loader = accelerator.prepare(DataLoader(
                SimCLRDatasetWrapper(dataset, 0, 0, 0, eval_mode=True), batch_size=eval_batch_size,
                collate_fn=collate_fn, sampler=test_subsampler
            ))
        else:
            train_loader = accelerator.prepare(DataLoader(
                dataset, batch_size=batch_size,
                collate_fn=collate_fn, sampler=train_subsampler
            ))

            valid_loader = accelerator.prepare(DataLoader(
                dataset, batch_size=eval_batch_size,
                collate_fn=collate_fn, sampler=valid_subsampler
            ))

            test_loader = accelerator.prepare(DataLoader(
                dataset, batch_size=eval_batch_size,
                collate_fn=collate_fn, sampler=test_subsampler
            ))


        model = model.to(device)

        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate)

        model, opt = accelerator.prepare(model, opt)
        
        test_auc, test_acc, test_rmse = model_train(fold, model, accelerator, opt, train_loader, valid_loader, test_loader, config)
    
        test_aucs.append(test_auc)
        test_accs.append(test_acc)
        test_rmses.append(test_rmse)
    
    test_auc = np.mean(test_aucs)
    test_acc = np.mean(test_accs)
    test_rmse = np.mean(test_rmses)

    now = (datetime.now()+timedelta(hours=9)).strftime("%Y%m%d-%H%M%S")  # KST time
    
    log_out_path = os.path.join(os.path.join("logs", "5-fold-cv", "{}".format(data_name)))
    os.makedirs(log_out_path, exist_ok=True)
    with open(os.path.join(log_out_path, "{}-{}".format(model_name, now)), 'w') as f:
        f.write("AUC\tACC\tRMSE\n")
        f.write("{:.5f}\t{:.5f}\t{:.5f}".format(test_auc, test_acc, test_rmse))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="dkt",
        help="The name of the model to train. \
            The possible models are in [dkt, dkvmn, sakt, akt]. \
            The default model is dkt."
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="assistments09",
        help="The name of the dataset to use in training. \
            The possible datasets are in [assistments09, assistments12, slepemapy]. \
            The default dataset is assistments2009. Other datasets will be updated."
    )
    args = parser.parse_args()

    config_file = open('configs/example.yaml', 'r')
    config = yaml.safe_load(config_file)
    config["model_name"] = args.model_name
    config["data_name"] = args.data_name
    main(config)
