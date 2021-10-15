import jammy.image as jimg
import jamtorch.prototype as jampt
import matplotlib.pyplot as plt
from jamtorch.trainer import step_lr
from jamtorch.utils import no_grad_func
from torch.optim.lr_scheduler import StepLR

import utils.diagnosis as dgns
from utils.scalars import scalar_helper
from viz.ps import seqSample2img, viz_sample

# pylint: disable=unused-argument


def epoch_start(trainer):
    dataset = trainer.train_loader.dataset
    trainer.test_sample = dataset.dataset.data[:50000]
    viz_sample(trainer.test_sample, "ground sample", "GT-sample.png")


def epoch_after_wrapper(cfg):
    @no_grad_func
    def check_gtsample_traj(trainer):
        model = trainer.model
        gt_sample = jampt.from_numpy(trainer.test_sample)
        z = dgns.forward_x2z(model, gt_sample, *scalar_helper(model))
        viz_sample(
            z, "test forward", f"forward_{trainer.epoch_cnt:02}.png", fix_lim=False
        )
        x = dgns.backward_z2x(model, z, *scalar_helper(model))
        viz_sample(x, "test backward", f"backward_{trainer.epoch_cnt:02}.png")

    @no_grad_func
    def epoch_after(trainer):
        sample = trainer.model.sample(50000)
        viz_sample(sample, "sde sample", f"epoch_{trainer.epoch_cnt:02}.png")
        check_gtsample_traj(trainer)

    return epoch_after


def check_process(cfg):
    def _fn(trainer):
        model = trainer.model
        gt_sample = jampt.from_numpy(trainer.test_sample)
        noise = model.sample_noise(5000)
        imgs = []

        # GT Forward
        f_process = dgns.forward_whole_process(model, gt_sample, *scalar_helper(model))
        fig = seqSample2img(f_process["data"], 10)
        fig.suptitle("GT Forward")
        imgs.append(jimg.plt2pil(fig))
        plt.close(fig)

        # GT Backward
        gt_noise = f_process["data"][-1]
        b_process = dgns.backward_whole_process(model, gt_noise, *scalar_helper(model))
        fig = seqSample2img(b_process["data"][::-1], 10)
        fig.suptitle("GT Backward")
        imgs.append(jimg.plt2pil(fig))
        plt.close(fig)

        # Noise Backward
        b_process = dgns.backward_whole_process(model, noise, *scalar_helper(model))
        fig = seqSample2img(b_process["data"], 10)
        fig.suptitle("Noise Backward")
        imgs.append(jimg.plt2pil(fig))
        plt.close(fig)

        # Deterministic Backward
        d_process = dgns.backward_deterministic_process(
            model, noise, *scalar_helper(model)
        )
        fig = seqSample2img(d_process["data"], 10)
        fig.suptitle("Deterministic Backward")
        imgs.append(jimg.plt2pil(fig))
        plt.close(fig)

        # LGV
        lgv_process = dgns.langevin_process(
            model, noise, -1, 1000, 0.05, model.condition, all_img=True
        )
        fig = seqSample2img(lgv_process["data_mean"], 10)
        fig.suptitle("LGV process")
        imgs.append(jimg.plt2pil(fig))
        plt.close(fig)

        fig = jimg.imgstack(imgs)
        jimg.savefig(fig, f"whole-{trainer.epoch_cnt:02}.png")
        plt.close(fig)

    return _fn


def points_trainer_register(trainer, cfg):
    trainer.register_event("val:end", epoch_after_wrapper(cfg))
    trainer.register_event("epoch:end", check_process(cfg))
    trainer.register_event("epoch:start", epoch_start)

    scheduler = StepLR(trainer.optimizer, step_size=2, gamma=cfg.optimizer.gamma)
    trainer.lr_scheduler = scheduler
    trainer.register_event("epoch:after", step_lr)
