import attr
import numpy as np
import torch
from jammy import jam_instantiate
from scipy.interpolate import interp1d

# pylint: disable=attribute-defined-outside-init, too-few-public-methods


def create_alpha_schedule(num_steps=100, t_start=0.0001, t_end=0.02):
    betas = np.linspace(t_start, t_end, num_steps)
    result = [1.0]
    alpha = 1.0
    for beta in betas:
        alpha *= 1 - beta
        result.append(alpha)
    return torch.FloatTensor(result)


def timestamp_fn(num_steps=100, t_start=0.0001, t_end=0.02):
    betas = np.linspace(t_start, t_end, num_steps)
    low_tri = np.tril(np.ones((num_steps, num_steps)))
    m = np.concatenate((np.zeros(num_steps).reshape(1, -1), low_tri), axis=0)
    times = m @ betas.reshape(-1, 1)
    assert times.size == num_steps + 1
    return torch.FloatTensor(times).flatten()


def diffusion_fn(num_steps=100, t_start=0.0001, t_end=0.02):
    betas = np.linspace(t_start, t_end, num_steps + 1)
    return torch.sqrt(torch.FloatTensor(betas))


def squareliner_fn(num_steps=100, t_start=0.0001, t_end=0.02):
    square = np.linspace(np.sqrt(t_start), np.sqrt(t_end), num_steps + 1)
    return torch.pow(torch.FloatTensor(square), 2)


def linear_fn(num_steps=100, t_start=0.0001, t_end=0.02):
    square = np.linspace(t_start, t_end, num_steps + 1)
    return torch.FloatTensor(square)


def exp_fn(num_steps=100, t_start=0.0001, t_end=0.02, exp=0.9):
    base = np.linspace(t_start ** exp, t_end ** exp, num_steps + 1)
    return torch.pow(torch.FloatTensor(base), 1.0 / exp)


def g_square_fn(num_steps=100, t_start=0.0001, t_end=0.02, exp=0.9):
    avg = (t_end - t_start) / num_steps
    time = exp_fn(num_steps, t_start, t_end, exp)
    dt = time[1:] - time[:-1]
    g = (avg / dt) ** 1.5
    return torch.cat([g[-1:] * 1.2, g])


@attr.s
class ExpTimer:
    num_steps = attr.ib(100)
    t_start = attr.ib(0.0001)
    t_end = attr.ib(0.02)
    exp = attr.ib(0.5)

    def __attrs_post_init__(self):
        self.base = torch.linspace(
            self.t_start ** self.exp, self.t_end ** self.exp, self.num_steps
        )
        self.fix_x_slot = torch.linspace(
            self.t_start ** self.exp, self.t_end ** self.exp, self.num_steps + 1
        )
        self.intervals = self.base[1:] - self.base[:-1]

    def __call__(self):
        value = torch.pow(self.fix_x_slot, 1.0 / self.exp)
        return self.deal_flip(value)

    def deal_flip(self, value):
        if self.exp > 1.0:
            value = self.t_start + self.t_end - value
            value = torch.flip(value, (0,))
        return value

    def rand(self):
        ratio = torch.rand(self.num_steps - 1)
        mid_times = ratio * self.intervals + self.base[:-1]
        times = torch.cat([self.base[:1], mid_times, self.base[-1:]]).flatten()
        value = torch.pow(times, 1.0 / self.exp)
        return self.deal_flip(value)

    def index(self, time):
        if np.isclose(self.t_start, self.t_end):
            return torch.pow(self.base[-1], 1.0 / self.exp) * torch.ones_like(time)
        time = torch.clip(time, self.t_start, self.t_end)
        return time
        # base = time ** self.exp
        # ratio = (base - base[0]) / (base[-1] - base[0])
        # times = ratio * (self.base[-1] - self.base[0]) + self.base[0]
        # return torch.pow(times, 1.0/ self.exp)


@attr.s
class SCurve:
    num_steps = attr.ib(100)
    t_start = attr.ib(0.0001)
    t_end = attr.ib(0.02)
    exp = attr.ib(0.5)

    def __attrs_post_init__(self):
        avg = (self.t_end - self.t_start) / self.num_steps
        self.ratio_x = [0.0, 0.2, 0.9, 1]
        # self.ratio_y = [0., 0.1, 0.1, 1]
        self.ratio_y = [0.0, avg, avg, 0.2]
        int_y = np.interp(
            np.linspace(0, 1, self.num_steps + 1), self.ratio_x, self.ratio_y
        )
        # delta = int_y * avg / 0.1
        delta = int_y
        self._time = np.cumsum(delta) + self.t_start

    def __call__(self):
        return torch.from_numpy(self._time).float()

    def rand(self):
        midpoints = np.linspace(0, 1, self.num_steps)
        delta_t = midpoints[1:] - midpoints[:-1]
        ratio = np.random.rand(self.num_steps - 1)
        mid_timestamps = delta_t * ratio + midpoints[:-1]
        timestamps = np.concatenate([[0], mid_timestamps, [1]]).flatten()
        return torch.from_numpy(
            np.interp(timestamps, np.linspace(0, 1, self.num_steps + 1), self._time)
        ).float()

    def index(self, time):
        if np.isclose(self.t_start, self.t_end):
            return torch.ones_like(time) * self.t_start
        return time


@attr.s
class FTimer:
    num_steps = attr.ib(100)
    t_start = attr.ib(0.0001)
    t_end = attr.ib(0.02)
    exp = attr.ib(0.5)

    def __attrs_post_init__(self):
        first_p = int(self.num_steps * 0.9)
        self.t1 = ExpTimer(first_p, self.t_start, self.t_end, self.exp)
        self.t2 = ExpTimer(self.num_steps - first_p, self.t_end, 0.5, 1.0 / self.exp)

    def __call__(self):
        return torch.cat([self.t1()[:-1], self.t2()])


@attr.s
class STimer:  # pylint: disable= too-many-instance-attributes
    num_steps = attr.ib(50)
    t_start = attr.ib(0.0001)
    t_end = attr.ib(0.02)
    up = attr.ib(True)  # pylint: disable= invalid-name

    def __attrs_post_init__(self):
        if self.up:
            x = [0, 0.1, 0.20, 0.4, 0.70, 0.9, 1.0]
            y = [1e-4, 0.05, 0.10, 0.6, 0.93, 0.98, 1.0]
        else:
            x = [0, 0.1, 0.30, 0.6, 0.93, 0.98, 1.0]
            y = [1e-4, 0.2, 0.35, 0.4, 0.70, 0.9, 1.0]
        self.interp1d_fn = interp1d(x, y, kind="cubic")
        fix_x_slot = np.linspace(1e-4, 1.0, self.num_steps + 1)
        dt = self.interp1d_fn(fix_x_slot)
        dt_sum = np.sum(dt) + self.t_start
        self.scale = self.t_end / dt_sum
        self.fix_t_slot = (np.cumsum(dt) + self.t_start) * self.scale
        self.time_fn = interp1d(fix_x_slot, self.fix_t_slot)
        self.fix_x_slot = fix_x_slot

        # for dealing with t
        random_x_slot = np.linspace(1e-4, 1.0, self.num_steps)
        self.random_t_slot = torch.from_numpy(self.time_fn(random_x_slot)).float()
        self.random_t_interval = self.random_t_slot[1:] - self.random_t_slot[:-1]

    def __call__(self):
        return torch.from_numpy(self.fix_t_slot).float()

    def rand(self):
        ratio = torch.rand(self.num_steps - 1)
        mid_times = ratio * self.random_t_interval + self.random_t_slot[:-1]
        return torch.cat(
            [self.random_t_slot[:1], mid_times, self.random_t_slot[-1:]]
        ).flatten()

    def index(self, time):
        if np.isclose(self.t_start, self.t_end):
            return self.t_start * torch.ones_like(time)
        time = torch.clip(time, self.t_start, self.t_end)
        return time


def instantiate_scaler(cfg):
    timer = jam_instantiate(cfg.model.time_fn)
    differ = jam_instantiate(cfg.model.diff_fn)
    conder = jam_instantiate(cfg.model.cond_fn)
    return timer, differ, conder


def scalar_helper(model):
    return model.timestamps, model.diffusion, model.condition
