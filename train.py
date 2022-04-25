import pandas as pd
import numpy as np
import torch
import os
import glob

from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


def model_train(
    fold,
    model,
    accelerator,
    opt,
    train_loader,
    valid_loader,
    test_loader,
    config,
    n_gpu,
    early_stop=True,
):
    train_losses = []
    avg_train_losses = []
    best_valid_auc = 0

    logs_df = pd.DataFrame()
    num_epochs = config["train_config"]["num_epochs"]
    model_name = config["model_name"]
    data_name = config["data_name"]
    train_config = config["train_config"]
    log_path = train_config["log_path"]

    now = (datetime.now() + timedelta(hours=9)).strftime("%Y%m%d-%H%M%S")  # KST time

    token_cnts = 0
    label_sums = 0
    for i in range(1, num_epochs + 1):
        for batch in tqdm(train_loader):
            opt.zero_grad()

            model.train()
            out_dict = model(batch)

            if n_gpu > 1:
                loss, token_cnt, label_sum = model.module.loss(batch, out_dict)
            else:
                loss, token_cnt, label_sum = model.loss(batch, out_dict)

            accelerator.backward(loss)

            token_cnts += token_cnt
            label_sums += label_sum

            if train_config["max_grad_norm"] > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=train_config["max_grad_norm"]
                )

            opt.step()
            train_losses.append(loss.item())

        print("token_cnts", token_cnts, "label_sums", label_sums)

        total_preds = []
        total_trues = []

        with torch.no_grad():
            for batch in valid_loader:
                model.eval()

                out_dict = model(batch)
                pred = out_dict["pred"].flatten()
                true = out_dict["true"].flatten()
                mask = true > -1
                pred = pred[mask]
                true = true[mask]

                total_preds.append(pred)
                total_trues.append(true)

            total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
            total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)

        valid_auc = roc_auc_score(y_true=total_trues, y_score=total_preds)

        path = os.path.join("saved_model", model_name, data_name)
        if not os.path.isdir(path):
            os.makedirs(path)

        if valid_auc > best_valid_auc:

            path = os.path.join(
                os.path.join("saved_model", model_name, data_name), "params_*"
            )
            for _path in glob.glob(path):
                os.remove(_path)
            best_valid_auc = valid_auc
            best_epoch = i
            torch.save(
                {"epoch": i, "model_state_dict": model.state_dict(),},
                os.path.join(
                    os.path.join("saved_model", model_name, data_name),
                    "params_{}".format(str(best_epoch)),
                ),
            )
        if i - best_epoch > 10:
            break

        # clear lists to track next epochs
        train_losses = []
        valid_losses = []

        total_preds, total_trues = [], []

        # evaluation on test dataset
        with torch.no_grad():
            for batch in test_loader:

                model.eval()

                out_dict = model(batch)

                pred = out_dict["pred"].flatten()
                true = out_dict["true"].flatten()
                mask = true > -1
                pred = pred[mask]
                true = true[mask]

                total_preds.append(pred)
                total_trues.append(true)

            total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
            total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

        test_auc = roc_auc_score(y_true=total_trues, y_score=total_preds)

        print(
            "Fold {}:\t Epoch {}\t\tTRAIN LOSS: {:.5f}\tVALID AUC: {:.5f}\tTEST AUC: {:.5f}".format(
                fold, i, train_loss, valid_auc, test_auc
            )
        )
    checkpoint = torch.load(
        os.path.join(
            os.path.join("saved_model", model_name, data_name),
            "params_{}".format(str(best_epoch)),
        )
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    total_preds, total_trues = [], []
    total_q_embeds, total_qr_embeds = [], []
    # evaluation on test dataset
    with torch.no_grad():
        for batch in test_loader:

            model.eval()

            out_dict = model(batch)

            pred = out_dict["pred"].flatten()
            true = out_dict["true"].flatten()
            mask = true > -1
            pred = pred[mask]
            true = true[mask]
            total_preds.append(pred)
            total_trues.append(true)

        total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
        total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

    auc = roc_auc_score(y_true=total_trues, y_score=total_preds)
    acc = accuracy_score(y_true=total_trues >= 0.5, y_pred=total_preds >= 0.5)
    rmse = np.sqrt(mean_squared_error(y_true=total_trues, y_pred=total_preds))

    print(
        "Best Model\tTEST AUC: {:.5f}\tTEST ACC: {:5f}\tTEST RMSE: {:5f}".format(
            auc, acc, rmse
        )
    )

    logs_df = logs_df.append(
        pd.DataFrame(
            {"EarlyStopEpoch": best_epoch, "auc": auc, "acc": acc, "rmse": rmse},
            index=[0],
        ),
        sort=False,
    )

    log_out_path = os.path.join(log_path, data_name)
    os.makedirs(log_out_path, exist_ok=True)
    logs_df.to_csv(
        os.path.join(log_out_path, "{}_{}.csv".format(model_name, now)), index=False
    )

    return auc, acc, rmse
