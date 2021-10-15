import torch
from jamtorch import as_cpu, no_grad_func

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals


@no_grad_func
def forward_whole_process(model, x, timestamps, diffusion, condition):
    x = x.clone()  # avoid overwrite the origin data
    # TODO: dict can be slow
    rtn = {"data": [x.clone()], "grad": [], "noise": [], "step_size": []}
    delta_t = timestamps[1:] - timestamps[:-1]
    for i_th, cur_delta_t in enumerate(delta_t):
        cond_f, diff_f = condition[i_th], diffusion[i_th]
        grad = model.drift(x, cond_f)
        grad_step = grad * cur_delta_t
        noise = torch.sqrt(cur_delta_t) * diff_f * model.sample_noise(x.shape[0])
        x += grad_step + noise
        rtn["data"].append(x.clone())
        rtn["grad"].append(grad)
        rtn["noise"].append((noise / cur_delta_t))
        rtn["step_size"].append(cur_delta_t)

    return rtn


@no_grad_func
def backward_whole_process(
    model, z, timestamps, diffusion, condition, drift_only=False, score_only=False
):
    rtn = {
        "data": [z.clone()],
        "grad": [],
        "noise": [],
        "drift": [],
        "diff": [],
        "score": [],
        "noise_step": [],
        "step_size": [],
    }
    z = z.clone()
    delta_t = timestamps[1:] - timestamps[:-1]
    for i_th, cur_delta_t in enumerate(torch.flip(delta_t, (0,))):
        cond_b, diff_b = condition[-i_th - 1], diffusion[-i_th - 1]
        drift = model.drift(z, cond_b)
        score = model.score(z, cond_b)
        diff = -(diff_b ** 2) * score
        if drift_only:
            grad = drift
        else:
            grad = drift + diff
        if score_only:
            grad = diff
        grad_step = grad * cur_delta_t
        noise = torch.sqrt(cur_delta_t) * diff_b * model.sample_noise(z.shape[0])
        z_mean = z - grad_step
        z = z_mean + noise
        rtn["data"].append(z_mean.clone())
        rtn["grad"].append(grad)
        rtn["drift"].append(drift)
        rtn["diff"].append(diff)
        rtn["score"].append(-score)
        rtn["noise"].append(noise / cur_delta_t)
        rtn["step_size"].append(cur_delta_t)
    return rtn


def fb_whole_process(model, x, timestamps, diffusion, condition, is_gt=True):
    f_process = forward_whole_process(model, x, timestamps, diffusion, condition)
    z = f_process["data"][-1]
    if not is_gt:
        z = torch.randn_like(z)
    b_process = backward_whole_process(model, z, timestamps, diffusion, condition)
    return f_process, b_process


# def fb_whole_process(model, x, timestamps, diffusion, condition, is_gt=True):
#     f_process = forward_whole_process(model, x, timestamps, diffusion, condition)
#     z = f_process["data"][-1]
#     if not is_gt:
#         z = torch.randn_like(z)
#     b_process = backward_whole_process(model, z, timestamps, diffusion, condition)

#     # convert data device and follow the same order
#     f_process = as_cpu(f_process)
#     b_process = as_cpu(b_process)
#     for _, item in b_process.items():
#         item.reverse()

#     composite = {
#         "f_data": f_process["data"],
#         "b_data": b_process["data"],
#         "f_grad": f_process["grad"],
#         "b_grad": b_process["grad"],
#         "b_drift": b_process["drift"]
#     }
#     return composite


def ema_whole_process(model, load_ema_fn, z, timestamps, diffusion, condition):
    """Only used in check ema in cmp_fb_process"""
    z = torch.randn_like(z)
    non_ema = backward_whole_process(model, z, timestamps, diffusion, condition)
    non_ema = as_cpu(non_ema)

    load_ema_fn(model)
    ema = backward_whole_process(model, z, timestamps, diffusion, condition)
    ema = as_cpu(ema)

    composite = {
        "ema_x": ema["data"],
        "non_ema_x": non_ema["data"],
        "ema_grad": ema["grad"],
        "non_ema_grad": non_ema["grad"],
        "ema_drift": ema["drift"],
        "non_ema_drift": non_ema["drift"],
    }
    return composite


@no_grad_func
def forward_data_process(model, x, timestamps, diffusion, condition):
    x = x.clone()  # avoid overwrite the origin data
    rtn = [x.clone()]
    delta_t = timestamps[1:] - timestamps[:-1]
    for i_th, cur_delta_t in enumerate(delta_t):
        cond_f, diff_f = condition[i_th], diffusion[i_th]
        grad = model.drift(x, cond_f)
        grad_step = grad * cur_delta_t
        noise = torch.sqrt(cur_delta_t) * diff_f * model.sample_noise(x.shape[0])
        x += grad_step + noise
        rtn.append(x.clone())

    return rtn


@no_grad_func
def forward_x2z(model, x, timestamps, diffusion, condition):
    x = x.clone()  # avoid overwrite the origin data
    delta_t = timestamps[1:] - timestamps[:-1]
    for i_th, cur_delta_t in enumerate(delta_t):
        cond_f, diff_f = condition[i_th], diffusion[i_th]
        grad = model.drift(x, cond_f)
        grad_step = grad * cur_delta_t
        noise = torch.sqrt(cur_delta_t) * diff_f * model.sample_noise(x.shape[0])
        x += grad_step + noise
    return x


@no_grad_func
def backward_z2x(model, z, timestamps, diffusion, condition):
    # from noise to data
    z = z.clone()
    if len(timestamps) < 2:
        return z
    delta_t = timestamps[1:] - timestamps[:-1]
    for i_th, cur_delta_t in enumerate(torch.flip(delta_t, (0,))):
        cond_b, diff_b = condition[-i_th - 1], diffusion[-i_th - 1]
        drift = model.drift(z, cond_b)
        score = model.score(z, cond_b)
        diff = -(diff_b ** 2) * score
        grad = drift + diff
        grad_step = grad * cur_delta_t
        noise = torch.sqrt(cur_delta_t) * diff_b * model.sample_noise(z.shape[0])
        z_mean = z - grad_step
        z = z_mean + noise
    return z_mean


def recon_x(model, x, timestamps, diffusion, condition):
    z = forward_x2z(model, x, timestamps, diffusion, condition)
    return backward_z2x(model, z, timestamps, diffusion, condition)


@no_grad_func
def backward_deterministic_process(model, z, timestamps, diffusion, condition):
    rtn = {
        "data": [z.clone()],
        "grad": [],
        "grad_step": [],
        "step_size": [],
    }
    z = z.clone()
    delta_t = timestamps[1:] - timestamps[:-1]
    for i_th, cur_delta_t in enumerate(torch.flip(delta_t, (0,))):
        cond_b, diff_b = condition[-i_th - 1], diffusion[-i_th - 1]
        grad = model.drift(z, cond_b) - 0.5 * diff_b ** 2 * model.score(z, cond_b)
        grad_step = grad * cur_delta_t
        z += -grad_step
        rtn["data"].append(z.clone())
        rtn["grad"].append(grad)
        rtn["grad_step"].append(grad_step)
        rtn["step_size"].append(cur_delta_t)
    return rtn


@no_grad_func
def langevin_process(model, z, idx, steps, snr, condition=None, all_img=False):
    condition = model.condition if condition is None else condition
    cond_b = condition[-idx - 1]
    if all_img:
        rtn = {
            "data": [z.clone()],
            "data_mean": [],
            "grad": [],
            "step_size": [],
        }
    z = z.clone()
    for _ in range(steps):
        noise = model.sample_noise(z.shape[0])
        grad = model.score(z, cond_b)
        noise_norm = torch.mean(torch.norm(noise.flatten(start_dim=1), dim=1))
        grad_norm = torch.mean(torch.norm(grad.flatten(start_dim=1), dim=1))

        step_size = (snr * noise_norm / grad_norm) ** 2 * 2

        z_mean = z + grad * step_size

        z = z_mean + torch.sqrt(2 * step_size) * noise

        if all_img:
            rtn["data"].append(z.clone())
            rtn["data_mean"].append(z_mean)
            rtn["grad"].append(grad)
            rtn["step_size"].append(step_size.item())

    if all_img:
        return rtn
    return z_mean
