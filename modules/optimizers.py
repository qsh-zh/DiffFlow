import torch
from jammy import jam_instantiate


def get_optimizer(opt_cfg, model):
    return jam_instantiate(opt_cfg, model.parameters())


def get_tune_optimizer(opt_cfg, model):
    return torch.optim.Adam(
        [
            {"params": model.drift.parameters(), "lr": opt_cfg.drift},
            {"params": model.score.parameters(), "lr": opt_cfg.score},
        ],
        lr=2e-4,
    )


def fix_drift_optimizer(opt_cfg, model):
    for param in model.drift.parameters():
        param.requires_grad = False
    return torch.optim.Adam(
        [{"params": model.score.parameters(), "lr": opt_cfg.score}], lr=2e-4
    )
