# pylint: skip-file
import jamtorch.ddp.ddp_utils as ddp_utils
import numpy as np
import torch
import torch.cuda.amp as amp
from jamtorch.utils.meta import as_float

# if ddp_utils.is_master():

#     def trainer_stat(trainer, stat):
#         trainer.cur_monitor.update(stat)


# else:

#     def trainer_stat(trainer, state):
#         pass


# trainer = None


class SdeF(torch.autograd.Function):
    @staticmethod
    @amp.custom_fwd
    def forward(ctx, x, model, timestamps, diffusion, condition, *model_parameter):
        shapes = [y0_.shape for y0_ in model_parameter]

        def _flatten(parameter):
            # flatten the gradient dict and parameter dict
            return torch.cat(
                [
                    param.flatten() if param is not None else x.new_zeros(shape.numel())
                    for param, shape in zip(parameter, shapes)
                ]
            )

        def _unflatten(tensor, length):
            # return object like parameter groups
            tensor_list = []
            total = 0
            for shape in shapes:
                next_total = total + shape.numel()
                # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
                tensor_list.append(
                    tensor[..., total:next_total].view((*length, *shape))
                )
                total = next_total
            return tuple(tensor_list)

        history_x_state = x.new_zeros(len(timestamps) - 1, *x.shape)
        rtn_logabsdet = x.new_zeros(x.shape[0])
        delta_t = timestamps[1:] - timestamps[:-1]
        new_x = x
        with torch.no_grad():
            for i_th, cur_delta_t in enumerate(delta_t):
                history_x_state[i_th] = new_x
                new_x, new_logabsdet = model.forward_step(
                    new_x,
                    cur_delta_t,
                    condition[i_th],
                    condition[i_th + 1],
                    diffusion[i_th],
                    diffusion[i_th + 1],
                )
                rtn_logabsdet += new_logabsdet
        ctx.model = model
        ctx._flatten = _flatten
        ctx._unflatten = _unflatten
        ctx.nparam = np.sum([shape.numel() for shape in shapes])
        ctx.save_for_backward(
            history_x_state.clone(), new_x.clone(), timestamps, diffusion, condition
        )
        return new_x, rtn_logabsdet

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, dL_dz, dL_logabsdet):
        history_x_state, z, timestamps, diffusion, condition = ctx.saved_tensors
        dL_dparameter = dL_dz.new_zeros((1, ctx.nparam))

        model, _flatten, _unflatten = ctx.model, ctx._flatten, ctx._unflatten
        model_parameter = tuple(model.parameters())
        delta_t = timestamps[1:] - timestamps[:-1]
        b_noise = {}
        with torch.no_grad():
            for bi_th, cur_delta_t in enumerate(torch.flip(delta_t, (0,))):
                bi_th += 1
                with torch.set_grad_enabled(True):
                    x = history_x_state[-bi_th].requires_grad_(True)
                    z = z.requires_grad_(True)
                    noise_b = model.cal_backnoise(
                        x, z, cur_delta_t, condition[-bi_th], diffusion[-bi_th]
                    )

                    cur_delta_s = -0.5 * (
                        torch.sum(noise_b.flatten(start_dim=1) ** 2, dim=1)
                    )
                    dl_dprev_state, dl_dnext_state, *dl_model_b = torch.autograd.grad(
                        (cur_delta_s),
                        (x, z) + model_parameter,
                        grad_outputs=(dL_logabsdet),
                        allow_unused=True,
                        retain_graph=True,
                    )
                    dl_dx, *dl_model_f = torch.autograd.grad(
                        (
                            model.cal_next_nodiffusion(
                                x, cur_delta_t, condition[-bi_th - 1]
                            )
                        ),
                        (x,) + model_parameter,
                        grad_outputs=(dl_dnext_state + dL_dz),
                        allow_unused=True,
                        retain_graph=True,
                    )
                    del x, z, dl_dnext_state
                b_noise[f"stat/{bi_th}"] = -1 * cur_delta_s.mean()
                z = history_x_state[-bi_th]
                dL_dz = dl_dx + dl_dprev_state
                dL_dparameter += _flatten(dl_model_b).unsqueeze(0) + _flatten(
                    dl_model_f
                ).unsqueeze(0)

            # trainer_stat(trainer, as_float(b_noise))

        return (dL_dz, None, None, None, None, *_unflatten(dL_dparameter, (1,)))
