import datasets.image_dataset as img_dataset
import datasets.points_dataset as ps_dataset

from .img_transform import *


def get_dataset(cfg):
    if cfg.dataset in ps_dataset.skd_func:
        return ps_dataset.get_ps_dataset(cfg)
    if cfg.dataset in ["MNIST", "CIFAR10"]:
        return img_dataset.get_img_dataset(cfg)

    raise RuntimeError("Not find dataset func")
