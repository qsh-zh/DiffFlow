from jammy.utils.meta import Singleton
from jamtorch.ddp.ema_trainer import EMATrainer
from jamtorch.trainer import LossException
from jamtorch.trainer.trainer_monitor import TrainerMonitor
from retry.api import retry_call


class Trainer(EMATrainer, metaclass=Singleton):
    def __init__(self, cfg, loss_fn):
        super().__init__(cfg, loss_fn)
        self.trainer_monitor = None

    def monitor_update(self):
        if self.trainer_monitor:
            self.trainer_monitor.update(
                {
                    **self.cur_monitor,  # pylint: disable=access-member-before-definition
                    "epoch": self.epoch_cnt,
                    "iter": self.iter_cnt,
                }
            )
        self.cur_monitor = dict()  # pylint: disable=attribute-defined-outside-init

    def set_monitor(self, is_wandb, tblogger=False):
        """
        docstring
        """
        self.trainer_monitor = TrainerMonitor(is_wandb, tblogger)

    def _impl_load_ckpt(self, state):
        # do not overwrite the time coef
        state["model"]["timestamps"] = self.model.module.timestamps
        state["model"]["diffusion"] = self.model.module.diffusion
        state["model"]["condition"] = self.model.module.condition
        state["model"]["delta_t"] = self.model.module.delta_t
        super()._impl_load_ckpt(state)

    def train_step(self, feed_dict):
        retry_call(
            super().train_step, fargs=[feed_dict], tries=3, exceptions=LossException
        )
