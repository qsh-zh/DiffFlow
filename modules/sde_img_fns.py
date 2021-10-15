import functools

import torch_fidelity
from jammy import io
from jamtorch import ddp, get_logger, no_grad_func
from jamtorch.data import get_batch, num_to_groups
from jamtorch.ddp import ddp_utils

from datasets import data_transform, inverse_data_transform
from utils.diagnosis import (
    backward_whole_process,
    backward_z2x,
    fb_whole_process,
    recon_x,
)
from utils.scalars import scalar_helper
from viz.img import (
    check_unnormal_imgs,
    save_seperate_imgs,
    tensor2imgnd,
    viz_img_process,
    wandb_write_ndimg,
)

logger = get_logger()


def image_fidelity(img_dir):
    metric = torch_fidelity.calculate_metrics(
        input1=img_dir,
        input2="cifar10-train",
        cuda=True,
        isc=False,
        fid=True,
        kid=False,
        verbose=False,
    )
    return metric["frechet_inception_distance"]


def epoch_start_wrapper(cfg):
    def _epoch_start(trainer):
        from viz.lines import check_dflow_coef

        model = trainer.mmodel
        dataset = trainer.train_loader.dataset
        num_grid = cfg.data.eval_n_samples
        sample = get_batch(dataset, num_grid * num_grid)

        trainer.test_sample = data_transform(cfg.data, sample)
        trainer.test_noise = model.sample_noise(num_grid * num_grid)

        check_unnormal_imgs(cfg, trainer.test_sample, num_grid, 0, "GT-sample")
        check_unnormal_imgs(cfg, trainer.test_noise, num_grid, 0, "GT-noise")

        check_dflow_coef(model)

    return _epoch_start


def viz_gt_process(model, x, gif_suffix, eval_n, reverse_transform_fn):
    forward_kv, backward_kv = fb_whole_process(
        model, x, model.timestamps, model.diffusion, model.condition, is_gt=True
    )
    viz_img_process(
        forward_kv,
        f"f_{gif_suffix}",
        eval_n,
        ["data", "grad", "noise"],
        reverse_transform_fn,
    )
    viz_img_process(
        backward_kv,
        f"b_{gif_suffix}",
        eval_n,
        ["data", "drift", "diff"],
        reverse_transform_fn,
    )
    return forward_kv, backward_kv


def viz_sample_process(model, z, gif_suffix, eval_n, reverse_transform_fn):
    backward_kv = backward_whole_process(
        model, z, model.timestamps, model.diffusion, model.condition
    )
    viz_img_process(
        backward_kv,
        f"s_{gif_suffix}",
        eval_n,
        ["data", "drift", "diff"],
        reverse_transform_fn,
    )
    b_imgnd = tensor2imgnd(
        reverse_transform_fn(backward_kv["data"][-1]), eval_n, eval_n
    )
    return backward_kv, b_imgnd


def epoch_after_wrapper(cfg):  # pylint: disable=too-many-statements
    eval_n = cfg.data.eval_n_samples
    reverse_transform_fn = functools.partial(inverse_data_transform, cfg.data)

    ## prepare fid check
    n_gpu = ddp_utils.get_world_size()
    img_per_gpu = cfg.data.fid.num_samples // n_gpu
    sample_fid_path = "sample_fid_imgs"
    io.makedirs(sample_fid_path)
    gtimg_per_gpu = cfg.data.val_size // n_gpu
    sample_gt_path = "fb_fid_imgs"
    io.makedirs(sample_gt_path)
    check_fid_fn = image_fidelity

    @ddp.master_only
    @no_grad_func
    def check_gtsample_traj(trainer):
        n_epoch, n_iter = trainer.epoch_cnt, trainer.iter_cnt

        model = trainer.ema.model
        gt_sample = trainer.test_sample.to(trainer.device)
        forward_kv, backward_kv = viz_gt_process(
            model, gt_sample, f"{n_epoch}_{n_iter}", eval_n, reverse_transform_fn
        )
        f_imgnd = tensor2imgnd(
            reverse_transform_fn(forward_kv["data"][-1]), eval_n, eval_n
        )
        b_imgnd = tensor2imgnd(
            reverse_transform_fn(backward_kv["data"][-1]), eval_n, eval_n
        )
        wandb_write_ndimg(f_imgnd, n_iter, "t_f")
        wandb_write_ndimg(b_imgnd, n_iter, "t_b")

    @ddp.master_only
    def check_sampling(trainer):
        logger.info(f"eval {trainer.iter_cnt}")
        n_epoch, n_iter = trainer.epoch_cnt, trainer.iter_cnt

        model = trainer.ema.model
        z = trainer.test_noise.to(trainer.device)
        _, sampling_img = viz_sample_process(
            model, z, f"{n_epoch}_{n_iter}", eval_n, reverse_transform_fn
        )
        wandb_write_ndimg(sampling_img, n_iter, "sample")

    @no_grad_func
    def runtime_fid_sample(trainer):
        model = trainer.ema.model
        batch_size = cfg.data.fid.batch_size
        cnt = trainer.rank * img_per_gpu
        for _batch_size in num_to_groups(img_per_gpu, batch_size):
            s_n = model.sample_noise(_batch_size)
            sample = backward_z2x(model, s_n, *scalar_helper(model))
            sample = inverse_data_transform(cfg.data, sample)
            save_seperate_imgs(sample.cpu(), sample_fid_path, cnt)
            cnt += _batch_size
        ddp_utils.barrier()

    @no_grad_func
    def runtime_fid_gt(trainer):
        model = trainer.ema.model
        cnt = trainer.rank * gtimg_per_gpu
        for data in trainer.val_loader:
            data = data[0].float().to(trainer.device)
            data_trans = data_transform(cfg.data, data)
            x = recon_x(
                model, data_trans, model.timestamps, model.diffusion, model.condition
            )
            data_rec = inverse_data_transform(cfg.data, x)
            save_seperate_imgs(data_rec, sample_gt_path, cnt)
            cnt += len(data)
        ddp_utils.barrier()

    @ddp.master_only
    def runtime_check_fid(trainer):
        sample_fid = check_fid_fn(sample_fid_path)
        gt_fid = check_fid_fn(sample_gt_path)
        print(sample_fid, gt_fid)
        trainer.cmdviz.update("eval", {"sample_fid": sample_fid, "gt_fid": gt_fid})

    @no_grad_func
    def epoch_after(trainer):
        ddp_utils.barrier()
        check_sampling(trainer)
        check_gtsample_traj(trainer)
        if cfg.model.enable_fid:
            runtime_fid_sample(trainer)
            runtime_fid_gt(trainer)
            runtime_check_fid(trainer)

    return epoch_after


def img_trainer_register(trainer, cfg):
    trainer.register_event("epoch:start", epoch_start_wrapper(cfg))
    trainer.register_event("val:start", epoch_after_wrapper(cfg))
