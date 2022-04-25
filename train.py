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
    # early_stopping = EarlyStopping(patience=7, verbose=True)
    # q_align_losses = []
    # q_unif_losses = []
    # i_align_losses = []
    # i_unif_losses = []

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
        # batch_q_align_losses = []
        # batch_q_unif_losses = []
        # batch_i_align_losses = []
        # batch_i_unif_losses = []
        for batch in tqdm(train_loader):
            opt.zero_grad()

            model.train()
            out_dict = model(batch)

            if n_gpu > 1:
                loss, token_cnt, label_sum = model.module.loss(batch, out_dict)
                # q_align_loss, i_align_loss, q_unif_loss, i_unif_loss = model.module.alignment_and_uniformity(out_dict)
            else:
                loss, token_cnt, label_sum = model.loss(batch, out_dict)
                # q_align_loss, i_align_loss, q_unif_loss, i_unif_loss = model.alignment_and_uniformity(out_dict)

            accelerator.backward(loss)

            token_cnts += token_cnt
            label_sums += label_sum

            if train_config["max_grad_norm"] > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=train_config["max_grad_norm"]
                )

            opt.step()
            train_losses.append(loss.item())

            # batch_q_align_losses.append(torch.mean(q_align_loss).item())
            # batch_q_unif_losses.append(torch.mean(q_unif_loss).item())
            # batch_i_align_losses.append(torch.mean(i_align_loss).item())
            # batch_i_unif_losses.append(torch.mean(i_unif_loss).item())

        print("token_cnts", token_cnts, "label_sums", label_sums)

        total_preds = []
        total_trues = []

        with torch.no_grad():
            for batch in valid_loader:
                model.eval()

                # print('batch', batch["questions"])
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

        # q_align_losses.append(np.average(batch_q_align_losses))
        # q_unif_losses.append(np.average(batch_q_unif_losses))
        # i_align_losses.append(np.average(batch_i_align_losses))

        # epoch_len = len(str(num_epochs))
        # print_msg = (f'[{i:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
        #                 f'train_loss: {train_loss:.5f} ' +
        #                 f'valid_loss: {valid_loss:.5f}')
        # print(print_msg)

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

        # total_unifs = []
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

    # best model이 저장되어 있는 last checkpoint를 로드함
    model.load_state_dict(checkpoint["model_state_dict"])

    # """Compute Question and Interaction Uniformity"""
    # q_unif = 0
    # i_unif = 0

    # if model_name == "akt":
    #     # print('q_embed', model.q_embed.weight.shape)
    #     # model.qr_embed
    #     q_unif = uniform_loss(model.q_embed.weight[1:,:]).detach().cpu().numpy()
    #     i_unif = uniform_loss(model.qr_embed.weight[1:,:]).detach().cpu().numpy()
    #     print('q_unif', q_unif)
    #     print('i_unif', i_unif)
    # if model_name == "cl4kt":
    #     # print('q_embed', model.module.question_embed.weight.shape)
    #     q_unif = uniform_loss(model.module.question_embed.weight[1:,:]).detach().cpu().numpy()
    #     i_unif = uniform_loss(model.module.interaction_embed.weight[1:,:]).detach().cpu().numpy()
    #     print('q_unif', q_unif)
    #     print('i_unif', i_unif)

    total_preds, total_trues = [], []
    total_q_embeds, total_qr_embeds = [], []
    # evaluation on test dataset
    with torch.no_grad():
        for batch in test_loader:

            model.eval()

            out_dict = model(batch)
            # batch_unif = out_dict["interaction_uniformity"].flatten()

            pred = out_dict["pred"].flatten()
            true = out_dict["true"].flatten()
            mask = true > -1
            pred = pred[mask]
            true = true[mask]
            total_preds.append(pred)
            total_trues.append(true)

            # if model_name == "akt" or model_name == "cl4kt":
            #     total_q_embeds.append(out_dict["q_embed"])
            #     total_qr_embeds.append(out_dict["qr_embed"])

        total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
        total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

        # if model_name == "akt" or model_name == "cl4kt":
        #     q_unif = uniform_loss(torch.cat(total_q_embeds)).detach().cpu().numpy()
        #     i_unif = uniform_loss(torch.cat(total_qr_embeds)).detach().cpu().numpy()

        # print('q_unif', q_unif)
        # print('i_unif', i_unif)

        # total_unifs = torch.cat(total_unifs).squeeze(-1).detach().cpu().numpy()

    # last_unif = np.average(total_unifs)

    # For AUC-calibration plot
    # pd.DataFrame({"total_true": total_trues, "total_pred": total_preds})\
    #     .to_csv("./auc-calibration/{}_{}_{}_true_pred.csv".format(data_name, model_name, fold), index=False, sep="\t")

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

    # pd.DataFrame({"q_align_loss": q_align_losses[:best_epoch-1], "q_unif_loss": q_unif_losses[:best_epoch-1],
    #               "i_align_loss": i_align_losses[:best_epoch-1], "i_unif_loss": i_unif_losses[:best_epoch-1]})\
    #     .to_csv("./logs/{}/align_unifom_fold_{}.csv".format(data_name, fold), index=False, sep='\t')
    # last_align = i_align_losses[:best_epoch-1][-1]
    # min_align = np.min(i_align_losses)
    # last_unif = i_unif_losses[:best_epoch-1][-1]
    # min_unif = np.min(i_unif_losses)

    return auc, acc, rmse


# def uniform_loss(x, t=2):
#     x = torch.nn.functional.normalize(x, dim=1)
#     return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
