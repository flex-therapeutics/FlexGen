import torch


def to_local(x, trans, rot):
    return torch.einsum("bnji,bnaj->bnai", rot, x - trans[..., None, :])


def to_global(x, trans, rot):
    return torch.einsum("bnij,bnaj->bnai", rot, x) + trans[..., None, :]


def endpoint_euler_step(state, endpoint, time, next_time):
    if not 0 <= time < next_time <= 1:
        raise ValueError("require 0 <= time < next_time <= 1")
    return state + (next_time - time) / (1 - time) * (endpoint - state)


def validate_joint_interpolant(config):
    if (
        list(config.t_interval) != [0.0, 1.0]
        or float(config.get("t_sampling_focus", 0)) != 0
    ):
        raise ValueError("Joint training supports uniform time on [0,1] only")
    if config.batch_ot != "none":
        raise ValueError(
            "Joint flow requires batch_ot:none; target-dependent prior alignment is unsupported"
        )
    if config.prior not in ("conditional", "pure_noise") or bool(
        config.get("anm_prior_scale", False)
    ):
        raise ValueError(
            "Unsupported joint prior/conditioning; only unscaled conditional or pure_noise supported"
        )
    if (
        config.rots.train_schedule != "linear"
        or config.rots.sample_schedule != "linear"
    ):
        raise ValueError(
            "Joint flow requires linear rotation corruption and integration"
        )
    if (
        config.trans.train_schedule != "linear"
        or config.trans.sample_schedule != "linear"
        or float(config.min_t) != 0
    ):
        raise ValueError(
            "Joint flow requires linear translations and exact t=0 prior start"
        )
