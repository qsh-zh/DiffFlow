import os.path as osp

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from jammy import stmap
from jammy.image import imwrite, nd2pil, plt2nd
from jammy.logging import Wandb
from torchvision.utils import make_grid, save_image

from datasets import inverse_data_transform

plt.switch_backend("agg")


def tensor2imgnd(tensor, n_rows, n_cols):  # pylint: disable=unused-argument
    grid = make_grid(tensor, n_rows)
    ndarr = (
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    return ndarr


def kv_img2gif(
    kv_tensor_imgs, fname, img_row, img_col, keys
):  # pylint: disable=too-many-locals
    save_imgs = []
    input_imgs = {key: torch.stack(kv_tensor_imgs[key]) for key in keys}
    length = min([input_imgs[key].shape[0] for key in keys])
    num_keys = len(keys)
    dpi = 128
    img_size = 400
    for i in range(length):
        fig, axs = plt.subplots(
            1, num_keys, figsize=(num_keys * img_size / dpi, img_size / dpi), dpi=dpi
        )
        for j in range(num_keys):
            cur_img = input_imgs[keys[j]][i]  # j-th key, i-th image
            img2show = tensor2imgnd(cur_img, img_row, img_col)
            axs[j].imshow(img2show)
            axs[j].axes.xaxis.set_visible(False)
            axs[j].axes.yaxis.set_visible(False)

        fig.suptitle(f"t={i:03d} {'  '.join(keys)}")
        save_imgs.append(np.asarray(plt2nd(fig)))
        plt.close()
    imageio.mimsave(f"{fname}.gif", save_imgs + ([save_imgs[-1]] * 5), fps=1)


def viz_img_process(procss_kv, gif_file, num_grid, keys, reverse_transform_fn):
    imgs = stmap(reverse_transform_fn, procss_kv)
    kv_img2gif(imgs, gif_file, num_grid, num_grid, list(keys))


def wandb_write_ndimg(img, epoch_cnt, naming):
    if Wandb.IS_ACTIVE:
        wandb.log(
            {naming: wandb.Image(nd2pil(img), caption=f"{naming}_{epoch_cnt:05}.png")}
        )
    imwrite(f"{naming}_{epoch_cnt:03}.png", img)


def save_seperate_imgs(sample, sample_path, cnt):
    batch_size = len(sample)
    for i in range(batch_size):
        save_image(sample[i], osp.join(sample_path, f"{cnt:07d}.png"))
        cnt += 1


def check_unnormal_imgs(cfg, x, num_grid, num_iter, fname):
    trans_x = inverse_data_transform(cfg.data, x)
    noise_nd = tensor2imgnd(trans_x, num_grid, num_grid)
    wandb_write_ndimg(noise_nd, num_iter, fname)
