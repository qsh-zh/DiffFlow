import numpy as np
import torch
from jamtorch.distributions import StandardNormal


def batch_noise_square(noise):
    return torch.sum(noise.flatten(start_dim=1) ** 2, dim=1)


# FIXME: cond_f, cond_b is scalar, could cause bugs

# pylint: disable=too-many-arguments


class BaseModel(torch.nn.Module):
    def __init__(self, data_shape, drift_net, score_net):
        super().__init__()
        self.data_shape = tuple(data_shape)
        self.drift = drift_net
        self.score = score_net
        self._distribution = StandardNormal([np.prod(data_shape)])

    def forward_step(self, x, step_size, cond_f, cond_b, diff_f, diff_b):
        forward_noise = self._distribution.sample(x.shape[0]).view(x.shape)
        z = (
            self.cal_next_nodiffusion(x, step_size, cond_f)
            + torch.sqrt(step_size) * diff_f * forward_noise
        )
        backward_noise = self.cal_backnoise(x, z, step_size, cond_b, diff_b)
        delta_s = -0.5 * (
            batch_noise_square(backward_noise) - batch_noise_square(forward_noise)
        )
        return z, delta_s

    def cal_backnoise(self, x, z, step_size, cond_b, diff_b):
        f_backward = self.drift(z, cond_b) - diff_b ** 2 * self.score(z, cond_b)
        return (x - z + f_backward * step_size) / (diff_b * torch.sqrt(step_size))

    def cal_forwardnoise(self, x, z, step_size, cond_f, diff_f):
        f_backward = self.drift(x, cond_f)
        return (z - x - f_backward * step_size) / (diff_f * torch.sqrt(step_size))

    def cal_next_nodiffusion(self, x, step_size, cond_f):
        return x + self.drift(x, cond_f) * step_size

    def cal_prev_nodiffusion(self, z, step_size, cond_b, diff_b):
        return (
            z
            - (self.drift(z, cond_b) - diff_b ** 2 * self.score(z, cond_b)) * step_size
        )

    def backward_step(self, z, step_size, cond_f, cond_b, diff_f, diff_b):
        backward_noise = self._distribution.sample(z.shape[0]).view(z.shape)
        x = (
            self.cal_prev_nodiffusion(z, step_size, cond_b, diff_b)
            + torch.sqrt(step_size) * diff_b * backward_noise
        )
        forward_noise = self.cal_forwardnoise(x, z, step_size, cond_f, diff_f)
        delta_s = -0.5 * (
            batch_noise_square(forward_noise) - batch_noise_square(backward_noise)
        )
        return x, delta_s

    def sample(self, num_samples, timestamps, diffusion, condition):
        z = self._distribution.sample(num_samples).view(-1, *self.data_shape)
        x, _ = self.backward(z, timestamps, diffusion, condition)
        return x

    def forward(self, x, timestamps, diffusion, condition):
        batch_size = x.shape[0]
        logabsdet = x.new_zeros(batch_size)
        delta_t = timestamps[1:] - timestamps[:-1]
        for i_th, cur_delta_t in enumerate(delta_t):
            x, new_det = self.forward_step(
                x,
                cur_delta_t,
                condition[i_th],
                condition[i_th + 1],
                diffusion[i_th],
                diffusion[i_th + 1],
            )
            logabsdet += new_det
        return x, logabsdet

    def backward(self, z, timestamps, diffusion, condition):
        delta_t = timestamps[1:] - timestamps[:-1]
        logabsdet = z.new_zeros(z.shape[0])
        for i_th, cur_delta_t in enumerate(torch.flip(delta_t, (0,))):
            z, new_det = self.backward_step(
                z,
                cur_delta_t,
                condition[-i_th - 2],
                condition[-i_th - 1],
                diffusion[-i_th - 2],
                diffusion[-i_th - 1],
            )
            logabsdet += new_det
        return z, logabsdet

    def forward_list(self, x):
        rtn = [x]
        for i_th, cur_delta_t in enumerate(self.delta_t):
            x, _ = self.forward_step(
                x,
                cur_delta_t,
                self.condition[i_th],
                self.condition[i_th + 1],
                self.diffusion[i_th],
                self.diffusion[i_th + 1],
            )
            rtn.append(x)
        return rtn

    def backward_list(self, z):
        rtn = [z]
        for i_th, cur_delta_t in enumerate(torch.flip(self.delta_t, (0,))):
            z, _ = self.backward_step(
                z,
                cur_delta_t,
                self.condition[-i_th - 2],
                self.condition[-i_th - 1],
                self.diffusion[-i_th - 2],
                self.diffusion[-i_th - 1],
            )
            rtn.append(z)
        return rtn
