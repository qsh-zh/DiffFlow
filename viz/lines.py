import matplotlib.pyplot as plt
import torch
from jammy.logging import wandb_plt
from jamtorch.utils import as_numpy


@wandb_plt
def draw_line(data, title, caption=None):
    fig, axs = plt.subplots(1, 1)
    axs.plot(data)
    fig.suptitle(title)
    fig.savefig(f"{title}.png")
    if caption is None:
        caption = title
    return fig, title


def check_dflow_coef(model, prefix_caption=None):
    if hasattr(model, "module"):
        model = model.module
    for name in ["timestamps", "diffusion", "condition", "delta_t"]:
        if hasattr(model, name):
            if prefix_caption is None:
                caption = name
            else:
                caption = f"{name}_{prefix_caption}"
            draw_line(as_numpy(getattr(model, name)), name, caption)


def plt_scalars(scalars, names):
    """viz scalars and names plot figure

    :param scalars: [description]
    :type scalars: List[Tensor,ndarray]
    :param names: names of scalars
    :type names: List[string]
    :return: plt,fig
    """
    length = len(names)
    if isinstance(scalars[0], torch.Tensor):
        scalars = as_numpy(scalars)
    fig, axs = plt.subplots(1, length, figsize=(length * 7, 1 * 7))
    for i_th, cur_data in enumerate(scalars):
        axs[i_th].plot(cur_data)
        axs[i_th].set_title(names[i_th])
    return fig


def plt_model_scalars(model):
    if isinstance(model, dict):
        timestamps = model["timestamps"].cpu().numpy()
        diffusion = model["diffusion"].cpu().numpy()
        condition = model["condition"].cpu().numpy()
    delta_t = timestamps[1:] - timestamps[:-1]
    scalars = [timestamps, diffusion, condition, delta_t]
    labels = ["timestamps", "diffusion", "condition", "delta_t"]
    return plt_scalars(scalars, labels)
