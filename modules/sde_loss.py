import numpy as np
from jammy import hyd_instantiate

from datasets import data_transform
from utils.scalars import instantiate_scaler
from viz.lines import check_dflow_coef

# pylint: disable=unused-argument, unused-variable


def img_preprocess(cfg, feed_dict, device):
    feed_dict = feed_dict[0].float().to(device)
    return data_transform(cfg.data, feed_dict)


def point_preprocess(cfg, feed_dict, device):
    return feed_dict.float().to(device)


def loss_fn_wrapper(cfg):
    preprocess_fn = hyd_instantiate(cfg.data.preprocess_fn, cfg)

    def loss_fn(trainer, feed_dict, is_train):
        model = trainer.mmodel
        feed_dict = preprocess_fn(feed_dict, trainer.device)
        z, logabsdet = model(feed_dict)

        norm_loss = -model.noise_log_prob(z.flatten(start_dim=1)).mean()
        det_loss = -logabsdet.mean()
        return (
            norm_loss + det_loss,
            {},
            {
                "norm": norm_loss,
                "det_loss": det_loss,
                "dim/norm": norm_loss / np.prod(z.shape[1:]),
            },
        )

    return loss_fn


def cont_loss_fn_wrapper(cfg):
    preprocess_fn = hyd_instantiate(cfg.data.preprocess_fn, cfg)
    n_idx = 0
    # FIXME: FIX IN CONFIG
    n_iters = np.array(cfg.model.N_iter)
    n_values = np.array(cfg.model.N_values)
    timer, differ, conder = instantiate_scaler(cfg)
    from jamtorch import get_logger

    logger = get_logger()

    def update_scalar(iter_cnt):
        nonlocal n_idx, timer, differ, conder
        cur_idx = np.sum(iter_cnt > n_iters)
        if cur_idx > n_idx:
            new_num = int(n_values[cur_idx])
            cfg.model.time_fn.num_steps = new_num
            cfg.model.diff_fn.num_steps = new_num
            cfg.model.cond_fn.num_steps = new_num
            n_idx = cur_idx
            timer, differ, conder = instantiate_scaler(
                cfg
            )  # pylint: disable=unused-variable
            logger.critical(f"\nIter{iter_cnt}: {n_idx} steps level: {new_num}")
            return True
        return False

    def loss_fn(trainer, feed_dict, is_train):
        model = trainer.mmodel
        if update_scalar(trainer.iter_cnt):
            model.timestamps = timer().to(trainer.device)
            model.diffusion = differ().to(trainer.device)
            model.condition = conder().to(trainer.device)
            model.delta_t = model.timestamps[1:] - model.timestamps[:-1]
            check_dflow_coef(model, prefix_caption=trainer.iter_cnt)
        feed_dict = preprocess_fn(feed_dict, trainer.device)

        cur_time = timer.rand()
        cur_diff = differ.index(cur_time).to(trainer.device)
        cur_cond = conder.index(cur_time).to(trainer.device)
        cur_time = cur_time.to(trainer.device)

        z, logabsdet = model.forward_cond(feed_dict, cur_time, cur_diff, cur_cond)

        norm_loss = -model.noise_log_prob(z.flatten(start_dim=1)).mean()
        det_loss = -logabsdet.mean()
        return (
            norm_loss + det_loss,
            {},
            {
                "norm": norm_loss,
                "det_loss": det_loss,
                "dim/norm": norm_loss / np.prod(z.shape[1:]),
            },
        )

    return loss_fn
