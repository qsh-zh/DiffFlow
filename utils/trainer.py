from jammy import Singleton
from jammy.utils.retry import retry_call
from jamtorch.trainer import LossException
from jamtorch.trainer.ema_trainer import EMATrainer


class Trainer(EMATrainer, metaclass=Singleton):
    def __init__(self, cfg, loss_fn):
        super().__init__(cfg, loss_fn)
        self.rank = 0
        self.is_master = True  # for sync ddp_trainer

    def _impl_load_ckpt(self, state):
        # do not overwrite the time coef
        state["model"]["timestamps"] = self.model.timestamps
        state["model"]["diffusion"] = self.model.diffusion
        state["model"]["condition"] = self.model.condition
        state["model"]["delta_t"] = self.model.delta_t
        super()._impl_load_ckpt(state)

    def train_step(self, feed_dict):
        retry_call(
            super().train_step, fargs=[feed_dict], tries=3, exceptions=LossException
        )
