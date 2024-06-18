# The Ornstein-Uhlenbeck process is a stochastic process that models the behavior
# of a particle undergoing Brownian motion with a linear drift term. It is often
# used in mathematical finance and physics to describe mean-reverting phenomena.
# The process is characterized by its speed of mean reversion, volatility, and time
# horizon.

from Vasicek import VasicekProcess

class OrnsteinUhlenbeckProcess(VasicekProcess):
    """
    Represents an Ornstein-Uhlenbeck process, which is a type of stochastic process
    used in mathematical finance and physics. It is a mean-reverting process that
    models the behavior of a particle undergoing Brownian motion with a linear
    drift term.

    Parameters:
    - speed (float): The speed at which the process reverts to its mean.
    - vol (float): The volatility of the process.
    - t (float): The time horizon of the process.
    - rng (RandomState): The random number generator to use.

    Inherits from the VasicekProcess class.

    Methods:
    - __str__(): Returns a string representation of the process.
    - __repr__(): Returns a string representation that can be used to recreate the process.
    """

    def __init__(self, speed=1, vol=1, t=1, rng=None):
        super().__init__(speed=speed, mean=0, vol=vol, t=t, rng=rng)

    def __str__(self):
        return "Ornstein-Uhlenbeck process with speed={s}, vol={v} on [0, {t}]".format(
            s=str(self.speed), v=str(self.vol), t=str(self.t)
        )

    def __repr__(self):
        return "OrnsteinUhlenbeckProcess(speed={s}, vol={v}, t={t})".format(
            s=str(self.speed), v=str(self.vol), t=str(self.t)
        )


