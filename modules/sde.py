from jammy import hyd_instantiate

from networks.diff_flow import DiffFlow, QuickDiffFlow

__all__ = ["init_model"]


def _init_model(cfg):
    if "_target_" in cfg:
        return hyd_instantiate(cfg)
    raise RuntimeError


def init_model(cfg):
    timestamps = hyd_instantiate(cfg.time_fn)()
    diffusion = hyd_instantiate(cfg.diff_fn)()
    condition = hyd_instantiate(cfg.cond_fn)()

    drift = _init_model(cfg.drift)
    score = _init_model(cfg.score)

    module = QuickDiffFlow if cfg.quick else DiffFlow
    return module(cfg.d_in, timestamps, diffusion, condition, drift, score)
