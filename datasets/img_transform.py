import torch

__all__ = ["logit_transform", "data_transform", "inverse_data_transform"]


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, x):
    if config.uniform_dequantization:
        x = x / 256.0 * 255.0 + torch.rand_like(x) / 256.0
    if config.gaussian_dequantization:
        x = x + torch.randn_like(x) * 0.01

    if config.rescaled:
        x = 2 * x - 1.0
    elif config.logit_transform:
        x = logit_transform(x)

    if config.image_mean is not None and config.image_std is not None:
        return (
            x - torch.FloatTensor(config.image_mean).to(x.device)[:, None, None]
        ) / torch.FloatTensor(config.image_std).to(x.device)[:, None, None]
    return x


def inverse_data_transform(config, x):
    if config.image_mean is not None and config.image_std is not None:
        x = (
            x * torch.FloatTensor(config.image_std).to(x.device)[:, None, None]
            + torch.FloatTensor(config.image_mean).to(x.device)[:, None, None]
        )

    if config.logit_transform:
        x = torch.sigmoid(x)
    elif config.rescaled:
        x = (x + 1.0) / 2.0

    return torch.clamp(x, 0.0, 1.0)
