import jammy.utils.imp as imp


def import_fns(cfg):
    model_fn = imp.load_class(f"modules.{cfg.name}.init_model")
    loss_fn = imp.load_class(cfg.loss_fn)
    register_fn = imp.load_class(cfg.trainer_register)
    return model_fn, loss_fn, register_fn
