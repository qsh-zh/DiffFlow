import os

import numpy as np
import torchvision.transforms as transforms
from jammy.utils.git import git_rootdir
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, LSUN, MNIST

from datasets.celeba import CelebA
from datasets.ffhq import FFHQ

__all__ = ["get_img_dataset"]


def mnist_dataset(data_path, img_size=28):
    train = MNIST(
        data_path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        ),
    )
    test = MNIST(
        data_path,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        ),
    )
    return train, test


def init_data_config(config):
    if "path" not in config:
        config.path = git_rootdir("data")


def get_img_dataset(config):  # pylint: disable=too-many-branches
    init_data_config(config)
    if config.dataset == "MNIST":
        return mnist_dataset(config.path, config.image_size)
    if config.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.image_size), transforms.ToTensor()]
        )

    if config.dataset == "CIFAR10":
        dataset = CIFAR10(
            os.path.join(config.path, "datasets", "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(config.path, "datasets", "cifar10_test"),
            train=False,
            download=True,
            transform=test_transform,
        )

    elif config.dataset == "CELEBA":
        if config.random_flip:
            dataset = CelebA(
                root=os.path.join(config.path, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        transforms.CenterCrop(140),
                        transforms.Resize(config.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )
        else:
            dataset = CelebA(
                root=os.path.join(config.path, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        transforms.CenterCrop(140),
                        transforms.Resize(config.image_size),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )

        test_dataset = CelebA(
            root=os.path.join(config.path, "datasets", "celeba_test"),
            split="test",
            transform=transforms.Compose(
                [
                    transforms.CenterCrop(140),
                    transforms.Resize(config.image_size),
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )

    elif config.dataset == "LSUN":
        train_folder = "{}_train".format(config.category)
        val_folder = "{}_val".format(config.category)
        if config.random_flip:
            dataset = LSUN(
                root=os.path.join(config.path, "datasets", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.image_size),
                        transforms.CenterCrop(config.image_size),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                    ]
                ),
            )
        else:
            dataset = LSUN(
                root=os.path.join(config.path, "datasets", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.image_size),
                        transforms.CenterCrop(config.image_size),
                        transforms.ToTensor(),
                    ]
                ),
            )

        test_dataset = LSUN(
            root=os.path.join(config.path, "datasets", "lsun"),
            classes=[val_folder],
            transform=transforms.Compose(
                [
                    transforms.Resize(config.image_size),
                    transforms.CenterCrop(config.image_size),
                    transforms.ToTensor(),
                ]
            ),
        )

    elif config.dataset == "FFHQ":
        if config.random_flip:
            dataset = FFHQ(
                path=os.path.join(config.path, "datasets", "FFHQ"),
                transform=transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()]
                ),
                resolution=config.image_size,
            )
        else:
            dataset = FFHQ(
                path=os.path.join(config.path, "datasets", "FFHQ"),
                transform=transforms.ToTensor(),
                resolution=config.image_size,
            )

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = (
            indices[: int(num_items * 0.9)],
            indices[int(num_items * 0.9) :],
        )
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)

    return dataset, test_dataset
