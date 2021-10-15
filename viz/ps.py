import matplotlib.pyplot as plt
import numpy as np
from jammy.io import get_name
from jammy.logging import wandb_plt

import datasets.points_dataset as ps_dataset


# pylint: disable=no-member
def fix_ax_lim(ax):
    ax.set_xlim(ps_dataset.DIM_LINSPACE[0], ps_dataset.DIM_LINSPACE[-1])
    ax.set_ylim(ps_dataset.DIM_LINSPACE[0], ps_dataset.DIM_LINSPACE[-1])


@wandb_plt
def viz_sample(sample, title_name, fig_name, sample_num=50000, fix_lim=True):
    sample = ps_dataset.restore(sample)
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].set_title(title_name)
    ax[0].hist2d(
        sample[:sample_num, 0],
        sample[:sample_num, 1],
        bins=ps_dataset.DIM_LINSPACE,
        cmap=plt.cm.jet,
    )
    ax[0].set_facecolor(plt.cm.jet(0.0))
    ax[1].plot(
        sample[:sample_num, 0],
        sample[:sample_num, 1],
        linewidth=0,
        marker=".",
        markersize=1,
    )
    if fix_lim:
        fix_ax_lim(ax[1])
    fig.suptitle(title_name)
    fig.savefig(fig_name)
    plt.axis("off")
    return fig, get_name(fig_name)


# @wandb_fig
# def check_density(density, title_name, fig_name):
#     global DIM_LINSPACE, G_MEAN, G_STD, G_SET_STD
#     sample = sample / G_SET_STD * G_STD + G_MEAN
#     fig, ax = plt.subplots(1, 1, figsize=(7, 7))
#     ax.set_title(title_name)
#     x = DIM_LINSPACE
#     yy, xx = np.meshgrid(x, x)
#     ax.pcolor(xx, yy, density.reshape([yy.shape[0], yy.shape[1]]))
#     fig.suptitle(title_name)
#     fig.savefig(fig_name)
#     return fig, get_name(fig_name)


# def plot_sample(sample, title_name, fig_name):
#     sample = ps_dataset.restore(sample)
#     fig, ax = plt.subplots(1, 1, figsize=(7, 7))
#     ax.hist2d(
#         sample[:, 0],
#         sample[:, 1],
#         bins=ps_dataset.DIM_LINSPACE,
#         cmap=plt.cm.jet,
#     )
#     ax.get_xaxis().set_ticks([])
#     ax.get_yaxis().set_ticks([])
#     fix_ax_lim(ax)
#     ax.set_facecolor(plt.cm.jet(0.0))
#     plt.savefig(fig_name)
#     plt.close()


# def plot_white_sample(sample, title_name, fig_name, sample_num=10000):
#     sample = ps_dataset.restore(sample)
#     fig, ax = plt.subplots(1, 1, figsize=(7, 7))
#     ax.plot(
#         sample[:sample_num, 0],
#         sample[:sample_num, 1],
#         linewidth=0,
#         marker=".",
#         markersize=1,
#         alpha=0.5,
#     )
#     ax.get_xaxis().set_ticks([])
#     ax.get_yaxis().set_ticks([])
#     fix_ax_lim(ax)
#     plt.savefig(fig_name)
#     plt.close()


def seqSample2img(list_x, n):
    length = len(list_x)
    idxes = np.linspace(0, length - 1, n, dtype=int)
    with plt.style.context("img"):
        fig, axs = plt.subplots(1, n, figsize=(3 * n, 3))
        for i_th, idx in enumerate(idxes):
            data = list_x[idx].cpu().numpy()
            axs[i_th].plot(
                data[:, 0], data[:, 1], linewidth=0, marker=".", markersize=1, alpha=0.5
            )
            axs[i_th].set_xlim(-2, 2)
            axs[i_th].set_ylim(-2, 2)
            axs[i_th].set_title(idx)
    return fig
