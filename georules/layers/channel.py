import numpy as np

from .base import Layer


class ChannelLayer(Layer):
    """Layer with fluvial channel geology."""

    def create_geology(self, channel_width, n_channels,
                       depth_width_ratio=0.4, friction_coeff=0.0009,
                       amplitude=10.0, slope=0.008, discharge=0.9,
                       meander_scale=0.8, avulsion_prob=0, poro_ave=0.3):
        """Generate fluvial channel geology.

        Parameters
        ----------
        channel_width : float
            Half-width of channel belt (same units as x_len/y_len).
        n_channels : int
            Number of channel generations.
        depth_width_ratio : float
            Channel depth-to-width ratio.
        friction_coeff : float
            Friction coefficient (Cf).
        amplitude : float
            Secondary flow amplitude (A). Controls lateral bank erosion
            that drives channel migration during the simulation.
        slope : float
            Channel slope (I).
        discharge : float
            Normalized discharge (Q).
        meander_scale : float
            Controls sinuosity of the initial channel path.  Internally
            mapped as ``s = 0.2 * meander_scale**2`` before being used
            as noise amplitude in the streamline generator.
            0 = perfectly straight, ~0.5 = mild meanders,
            ~1.0 = moderate meanders (default 0.8), ~2.0 = legacy-level
            sinuosity (equivalent to the original hard-coded s=0.8).
        avulsion_prob : float
            Probability of avulsion (0 to 1).
        poro_ave : float
            Reference porosity for channel fill (default 0.3).
        """
        from ._fluvial import fluvial

        self.poro_ave = poro_ave

        engine = fluvial(
            b=channel_width,
            nx=self.nx, ny=self.ny, nz=self.nz,
            xmn=self.dx / 2, ymn=self.dy / 2,
            xsiz=self.dx, ysiz=self.dy, zsiz=self.dz,
            dwratio=depth_width_ratio,
            Cf=friction_coeff, A=amplitude, I=slope, Q=discharge,
            meander_scale=meander_scale, pavul=avulsion_prob,
        )
        engine.simulation(nchannel=n_channels)

        self.poro_mat = engine.poro
        self.facies = engine.facies.astype(int)
        self.active = (self.facies > 0).astype(int)
        # Simple exponential poro-perm transform
        self.perm_mat = 10.0 * np.exp(20.0 * self.poro_mat) * self.active
