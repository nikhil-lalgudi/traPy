import numpy as np
from error_handler import check_numeric, check_positive_integer, check_numeric_or_single_arg_callable, ensure_single_arg_constant_function
from base import BaseTimeProcess

from diffusion import DiffusionProcess

class ExtendedVasicekProcess(DiffusionProcess):
    def __init__(self, speed=1, mean=0, vol=1, t=1, rng=None):
        super().__init__(speed=speed, mean=mean, vol=vol, volexp=0, t=t, rng=rng)

    def __str__(self):
        return "Extended Vasicek process with speed={s}, mean={m}, vol={v} on [0, {t}]".format(
            s=str(self.speed), m=str(self.mean), v=str(self.vol), t=str(self.t)
        )

    def __repr__(self):
        return "ExtendedVasicekProcess(speed={s}, mean={m}, vol={v}, t={t})".format(
            s=str(self.speed), m=str(self.mean), v=str(self.vol), t=str(self.t)
        )


class VasicekProcess(ExtendedVasicekProcess):
    def __init__(self, speed=1, mean=1, vol=1, t=1, rng=None):
        super().__init__(
            speed=ensure_single_arg_constant_function(speed),
            mean=ensure_single_arg_constant_function(mean),
            vol=ensure_single_arg_constant_function(vol),
            t=t,
            rng=rng,
        )

    def __str__(self):
        return "Vasicek process with speed={s}, mean={m}, vol={v} on [0, {t}]".format(
            s=str(self.speed), m=str(self.mean), v=str(self.vol), t=str(self.t)
        )

    def __repr__(self):
        return "VasicekProcess(speed={s}, mean={m}, vol={v}, t={t})".format(
            s=str(self.speed), m=str(self.mean), v=str(self.vol), t=str(self.t)
        )