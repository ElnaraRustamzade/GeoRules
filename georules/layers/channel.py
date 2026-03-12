import numpy as np

from .base import Layer


class ChannelLayerBase(Layer):
    """Base class for channel-type geological layers."""

    def _finalize_properties(self, engine_poro, engine_facies, poro_ave):
        """Convert engine output arrays into standard Layer properties.

        Shared by all channel subclasses to ensure consistent
        poro_mat, facies, active, perm_mat derivation.
        """
        self.poro_ave = poro_ave
        self.poro_mat = engine_poro
        self.facies = engine_facies.astype(int)
        self.active = (self.facies > 0).astype(int)
        self.perm_mat = 10.0 * np.exp(20.0 * self.poro_mat) * self.active


class MeanderingChannelLayer(ChannelLayerBase):
    """Layer with meandering fluvial channel geology."""

    def create_geology(self, channel_width, n_channels,
                       depth_width_ratio=0.4, friction_coeff=0.0009,
                       amplitude=10.0, slope=0.008, discharge=0.9,
                       meander_scale=0.8, avulsion_prob=0, poro_ave=0.3):
        """Generate meandering channel geology.

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
        self._finalize_properties(engine.poro, engine.facies, poro_ave)


class BraidedChannelLayer(ChannelLayerBase):
    """Layer with braided fluvial channel geology."""

    def create_geology(self, braidplain_width, n_channels, n_threads=3,
                       thread_width=None, depth_width_ratio=0.15,
                       slope=0.008, discharge=0.9,
                       bar_poro_factor=0.7, poro_ave=0.3):
        """Generate braided channel geology.

        Parameters
        ----------
        braidplain_width : float
            Total width of the braided belt (same units as x_len/y_len).
        n_channels : int
            Number of channel generations (aggradation steps).
        n_threads : int
            Number of simultaneous channel threads per generation.
        thread_width : float or None
            Half-width of each individual thread. If None,
            auto-calculated as braidplain_width / (2 * n_threads).
        depth_width_ratio : float
            Individual thread depth-to-width ratio (default 0.15,
            shallower than meandering default of 0.4).
        slope : float
            Channel slope.
        discharge : float
            Normalized discharge.
        bar_poro_factor : float
            Bar porosity as fraction of poro_ave (default 0.7).
        poro_ave : float
            Reference porosity for channel fill (default 0.3).
        """
        from ._braided import braided

        engine = braided(
            braidplain_width=braidplain_width,
            n_threads=n_threads,
            thread_width=thread_width,
            nx=self.nx, ny=self.ny, nz=self.nz,
            xmn=self.dx / 2, ymn=self.dy / 2,
            xsiz=self.dx, ysiz=self.dy, zsiz=self.dz,
            dwratio=depth_width_ratio,
            I=slope, Q=discharge,
            bar_poro_factor=bar_poro_factor,
        )
        engine.simulation(n_channels=n_channels)
        self._finalize_properties(engine.poro, engine.facies, poro_ave)

