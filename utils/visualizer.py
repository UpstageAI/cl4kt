import numpy as np
import matplotlib
import matplotlib.pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc("font", size=BIGGER_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def draw_heatmap(prob, questions, pred, gt, path, name=""):
    ## Draw heatmap from the estimations
    # prob: [L, Q]
    # questions: [L]
    # pred: [L]
    # gt: [L]

    fig, ax = plt.subplots(
        figsize=(30, 9), nrows=3, gridspec_kw={"height_ratios": [1, 4, 4]}
    )
    # prob = prob.transpose(0,1)
    print(prob)
    data_ = np.zeros([2, prob.shape[0]])
    data_[0, :] = gt
    data_[1, :] = pred
    # prob = np.concatenate( (data_.T, prob), axis=1)
    ax[0].imshow(data_, cmap=plt.get_cmap("bwr"), aspect="auto")
    ax[0].set_title("Knowledge Tracing {}".format(name))
    ax[0].set_ylabel("Pred-GT")
    # ax[0].figure.set_size_inches(30, 1)

    extent = (0, prob.shape[0], prob.shape[1], 0)

    prob = prob.T  # [Q, L]
    im = ax[1].imshow(prob, extent=extent, cmap=plt.get_cmap("bwr"), aspect="auto")
    ax[1].set_ylabel("Questions")
    ax[1].set_xlabel("Time")
    # for t in range(len(questions)):
    #     point = ax.text(t, questions[t], '*', ha="center", va="center", color="w")
    ax[1].grid(which="minor", color="gray", linewidth=0.5)
    # fig.tight_layout()

    question_order = np.argsort(prob.sum(-1))
    prob = prob[question_order]
    im = ax[2].imshow(prob, extent=extent, cmap=plt.get_cmap("bwr"), aspect="auto")
    ax[2].set_ylabel("Questions (Ordered)")
    ax[2].set_xlabel("Time")

    # fig.tight_layout()
    plt.savefig(path, dpi=300)
    # plt.show()
