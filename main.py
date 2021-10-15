import hydra
import jamtorch.prototype as jampt
import torch
import torch.multiprocessing as mp
from jammy import hydpath, jam_instantiate, link_hyd_run, load_class
from jammy.logging import Wandb, get_logger
from jamtorch.data import get_subset
from jamtorch.ddp import ddp_utils
from jamtorch.trainer import check_loss_error, trainer_save_cfg
from omegaconf import OmegaConf

from datasets import get_dataset
from modules import import_fns


def run(cfg):
    if ddp_utils.is_master():
        Wandb.launch(cfg, cfg.log, True)
        get_logger(
            "jam_.log",
            clear=True,
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
            level="DEBUG",
        )
    jampt.set_gpu_mode(cfg.cuda, cfg.trainer.gpu)

    init_model, loss_fn_wrapper, trainer_register = import_fns(cfg.model)

    trainer_str = (
        "utils.ddp_trainer.Trainer" if cfg.is_dist else "utils.trainer.Trainer"
    )
    trainer = load_class(trainer_str)(cfg.trainer, loss_fn_wrapper(cfg))
    model = init_model(cfg.model)
    optimizer = jam_instantiate(cfg.optimizer.fn, cfg.optimizer, model)
    trainer.set_model_optim(model, optimizer)
    trainer_register(trainer, cfg)
    check_loss_error(trainer)

    # data
    trainset, valset = get_dataset(cfg.data)
    trainset = get_subset(trainset, cfg.data.train_size)
    valset = get_subset(valset, cfg.data.val_size)
    train_loader, train_sampler, val_loader, val_sampler = jam_instantiate(
        cfg.data.dataloader,
        trainset,
        valset,
        rank=cfg.trainer.rank,
        world_size=cfg.trainer.world_size,
    )
    if cfg.is_dist:
        trainer.set_sampler(train_sampler, val_sampler)
    trainer.set_dataloader(train_loader, val_loader)

    if ddp_utils.is_master():
        trainer_save_cfg(trainer, cfg)
        trainer.set_monitor(cfg.log)
        trainer.save_ckpt()

    trainer.train()

    Wandb.finish()


@ddp_utils.ddp_runner
def mock_run(cfg):
    run(cfg)


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    link_hyd_run()
    cfg.data.path = hydpath("data")  # address hyd relative path
    if cfg.is_dist:
        world_size = torch.cuda.device_count()
        ddp_utils.prepare_cfg(cfg)
        mp.spawn(mock_run, args=(world_size, None, cfg), nprocs=world_size, join=True)
    else:
        run(cfg)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
