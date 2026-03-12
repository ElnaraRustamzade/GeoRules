"""Braided channel simulation engine (private module).

Generates a braided fluvial channel system within a braidplain
envelope.  Multiple narrow low-sinuosity channel threads are placed
at spread y-positions, some with bifurcation-bar-confluence (BBC)
splits.  Bars fill the gaps between channels using grid-based
scan-line rasterization.

Reuses genchannel() JIT functions from _genchannel.py for
thread-to-grid population.

Facies codes (ordered by reservoir quality):
    0 — shale (background, worst)
    1 — braid bar (intermediate)
    2 — active channel fill (best)
"""
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import binary_dilation
import scipy.signal

from ._genchannel import genchannel


class braided:
    """Braided fluvial channel simulation engine."""

    def __init__(self, braidplain_width, n_threads=3, thread_width=None,
                 nx=256, ny=128, nz=64,
                 xmn=8, ymn=8, xsiz=16, ysiz=16, zsiz=3,
                 dwratio=0.15, I=0.008, Q=0.9,
                 bar_poro_factor=0.7):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.xmn = xmn
        self.ymn = ymn
        self.xsiz = xsiz
        self.ysiz = ysiz
        self.zsiz = zsiz
        self.dwratio = dwratio
        self.I = I
        self.Q = Q
        self.bar_poro_factor = bar_poro_factor

        self.braidplain_width = braidplain_width
        self.n_threads = n_threads

        # Thread width: narrow enough that multiple channels are clearly
        # visible with bar islands between them.  Each channel carves
        # ~2*thread_width, so total channel = n_threads * 2 * tw.
        # Target: channels occupy ~30-50% of braidplain.
        # Enforce minimums so channels are both laterally visible (≥3.5
        # cells wide) and vertically deep enough (≥2 cells deep).
        if thread_width is None:
            min_lateral = 3.5 * max(xsiz, ysiz)
            min_depth = 2.0 * zsiz / dwratio   # ≥2 cells deep
            self.thread_width = max(braidplain_width / (5.0 * n_threads),
                                    min_lateral, min_depth)
        else:
            self.thread_width = thread_width

        # Grid bounds (same convention as fluvial engine)
        self.xmin = xmn - 0.5 * xsiz
        self.ymin = ymn - 0.5 * ysiz
        self.xmax = self.xmin + xsiz * nx
        self.ymax = self.ymin + ysiz * ny
        self.x = np.linspace(xmn, self.xmax - xmn, nx)
        self.y = np.linspace(ymn, self.ymax - ymn, ny)

        # Streamline generation parameters
        self.step0 = (xsiz + ysiz) * 8
        self.ndis0 = int((((self.xmax - self.xmin) +
                           (self.ymax - self.ymin)) / 2.0) /
                         ((xsiz + ysiz) / 2)) * 4

        # Channel depth in grid cells (min 2 so channels are visible)
        self.cz = max(2, int(self.dwratio * self.thread_width / self.zsiz))

        # Output arrays
        self.poro0 = 0.3
        self.poro = 0.05 * np.ones((nx, ny, nz))
        self.facies = np.zeros((nx, ny, nz))

    # ------------------------------------------------------------------
    # Streamline generation
    # ------------------------------------------------------------------

    def _generate_thread_streamline(self, y0, x0=-1000, s=0.05):
        """Generate a low-sinuosity streamline for a single thread.

        Returns
        -------
        cx, cy : ndarray or (None, None) if streamline invalid.
        """
        k = 0.1 * self.step0
        h = 0.8
        phi = np.arcsin(h)
        b1 = 2.0 * np.exp(-k * h) * np.cos(k * np.cos(phi))
        b2 = -1.0 * np.exp(-2.0 * k * h)
        mm0 = s * np.random.normal(0, 1, size=self.ndis0 + 40)
        mm0 = mm0[20:-20]
        ar = np.array([1, b1, b2])
        theta = scipy.signal.lfilter([1], ar, mm0)

        cx0 = np.cumsum(self.step0 * np.cos(theta)) + x0
        cy0 = np.cumsum(self.step0 * np.sin(theta)) + y0
        cx0 = np.append([x0], cx0)
        cy0 = np.append([y0], cy0)

        idx0 = np.arange(cx0.size)
        past_end = idx0[cx0 > self.xmax]
        if past_end.size == 0 or cx0.size < 20:
            return None, None
        ndis00 = past_end[0] + 4
        cx0 = cx0[:ndis00]
        cy0 = cy0[:ndis00]

        # Spline resample to fine spacing — ensure at least 3 points
        # per grid cell so BBC segments have enough points for
        # curvature fitting.
        length = np.zeros(cx0.size)
        length[1:] = np.sqrt((cx0[1:] - cx0[:-1])**2 +
                             (cy0[1:] - cy0[:-1])**2)
        length = np.cumsum(length)
        if length[-1] < 1e-6:
            return None, None
        splx = UnivariateSpline(length, cx0, k=5, s=0)
        sply = UnivariateSpline(length, cy0, k=5, s=0)
        target_density = 3.0 / min(self.xsiz, self.ysiz)
        n_fine = max(cx0.size * 18,
                     int(length[-1] * target_density))
        fine_length = np.linspace(0, length[-1], n_fine)
        cx = splx(fine_length)
        cy = sply(fine_length)

        return cx, cy

    def _constrain_to_braidplain(self, cy, center_y):
        """Clamp cy values to stay within braidplain envelope."""
        half_width = self.braidplain_width / 2.0
        np.clip(cy, center_y - half_width, center_y + half_width, out=cy)
        return cy

    # ------------------------------------------------------------------
    # Curvature computation
    # ------------------------------------------------------------------

    def _compute_curvature(self, cx, cy, width=None):
        """Compute curvature, thalweg, velocity vectors for a segment.

        Parameters
        ----------
        width : float or None
            Channel half-width for chwidth array.
        """
        if width is None:
            width = self.thread_width

        dlength = np.sqrt((cx[1:] - cx[:-1])**2 + (cy[1:] - cy[:-1])**2)
        dlength = np.append(0, dlength)
        cum_length = np.cumsum(dlength)

        if cum_length[-1] < 1e-6:
            return None, None, None, None, None, None, None, None

        splx = UnivariateSpline(cum_length, cx, k=3, s=0)
        sply = UnivariateSpline(cum_length, cy, k=3, s=0)

        step = cum_length[-1] / max(cx.size - 1, 1)
        nstep = int((cum_length[-1] - cum_length[0]) / step) + 1
        length = np.linspace(cum_length[0], cum_length[-1] - 0.1, nstep)

        cx_out = splx(length)
        cy_out = sply(length)

        # Smooth
        zz = 0.2
        cx_out[1:-1] = (cx_out[1:-1] * (1 - zz)
                        + zz / 2 * cx_out[:-2] + zz / 2 * cx_out[2:])
        cy_out[1:-1] = (cy_out[1:-1] * (1 - zz)
                        + zz / 2 * cy_out[:-2] + zz / 2 * cy_out[2:])

        s1 = UnivariateSpline(length, cx_out, k=5, s=4000)
        s2 = UnivariateSpline(length, cy_out, k=5, s=4000)

        vx_fn = s1.derivative(n=1)
        ax_fn = s1.derivative(n=2)
        vy_fn = s2.derivative(n=1)
        ay_fn = s2.derivative(n=2)

        vx = vx_fn(length)
        vy = vy_fn(length)
        denom = (vx**2 + vy**2)**1.5
        denom[denom == 0] = 1e-12
        curv = (vx * ay_fn(length) - ax_fn(length) * vy) / denom

        maxcurvr = max(curv.max(), 0) + 0.0001
        maxcurvl = max((-curv).max(), 0) + 0.0001
        thalweg = curv.copy()
        thalweg[curv >= 0] = 0.5 + curv[curv >= 0] * 0.25 / maxcurvr
        thalweg[curv < 0] = 0.5 + curv[curv < 0] * 0.25 / maxcurvl

        chwidth = width * np.ones(len(length))

        return cx_out, cy_out, vx, vy, curv, thalweg, chwidth, length

    # ------------------------------------------------------------------
    # BBC path builder
    # ------------------------------------------------------------------

    def _build_braided_path(self, trunk_cx, trunk_cy):
        """Build bifurcation-bar-confluence units along a trunk path.

        Most of the path length is in BBC mode (split into two
        sub-threads) with only brief convergence points between.

        Returns
        -------
        segments : list of (cx, cy, width) tuples
        bars : list of (left_cx, left_cy, right_cx, right_cy) tuples
        """
        segments = []
        bars = []

        n = trunk_cx.size
        if n < 30:
            segments.append((trunk_cx, trunk_cy, self.thread_width))
            return segments, bars

        # Arc-length parameterisation
        dx = np.diff(trunk_cx)
        dy = np.diff(trunk_cy)
        ds = np.sqrt(dx**2 + dy**2)
        cum_s = np.concatenate([[0], np.cumsum(ds)])
        total_s = cum_s[-1]

        # Tangent → perpendicular directions
        tangent_x = np.append(dx, dx[-1])
        tangent_y = np.append(dy, dy[-1])
        mag = np.sqrt(tangent_x**2 + tangent_y**2)
        mag[mag < 1e-12] = 1e-12
        perp_x = -tangent_y / mag
        perp_y = tangent_x / mag

        # Smooth perpendicular directions
        kern_size = min(21, max(3, n // 50))
        if kern_size % 2 == 0:
            kern_size += 1
        kernel = np.ones(kern_size) / kern_size
        if n > kern_size + 2:
            perp_x = np.convolve(perp_x, kernel, mode='same')
            perp_y = np.convolve(perp_y, kernel, mode='same')
            mag2 = np.sqrt(perp_x**2 + perp_y**2)
            mag2[mag2 < 1e-12] = 1e-12
            perp_x /= mag2
            perp_y /= mag2

        # Sub-thread width (narrower than trunk, but visible on grid)
        sub_width = max(0.6 * self.thread_width,
                        1.5 * max(self.xsiz, self.ysiz))

        # BBC parameters — long bifurcations, short gaps
        # bar_length: 8-15 × width → most of the path is bifurcated
        # spacing: 2-4 × width → brief convergence points
        # max_offset: 1.0-1.5 × width → sub-threads clearly separated
        current_s = np.random.uniform(1, 3) * self.thread_width
        prev_idx = 0

        while current_s < total_s - 2 * self.thread_width:
            bif_idx = int(np.searchsorted(cum_s, current_s))
            if bif_idx >= n - 10:
                break

            bar_len = np.random.uniform(8, 15) * self.thread_width
            conf_s = current_s + bar_len
            conf_idx = int(np.searchsorted(cum_s, conf_s))

            if conf_idx >= n - 5:
                break

            # Trunk segment before this BBC (brief convergence)
            if bif_idx > prev_idx + 10:
                segments.append((
                    trunk_cx[prev_idx:bif_idx + 1].copy(),
                    trunk_cy[prev_idx:bif_idx + 1].copy(),
                    self.thread_width,
                ))

            # BBC sub-threads via perpendicular sine offset
            sl = slice(bif_idx, conf_idx + 1)
            bbc_cx = trunk_cx[sl]
            bbc_cy = trunk_cy[sl]
            bbc_perp_x = perp_x[sl]
            bbc_perp_y = perp_y[sl]

            # Sine offset profile
            t_norm = np.linspace(0, 1, bbc_cx.size)
            max_offset = np.random.uniform(1.0, 1.5) * self.thread_width
            offset = max_offset * np.sin(np.pi * t_norm)

            left_cx = bbc_cx + bbc_perp_x * offset
            left_cy = bbc_cy + bbc_perp_y * offset
            right_cx = bbc_cx - bbc_perp_x * offset
            right_cy = bbc_cy - bbc_perp_y * offset

            segments.append((left_cx.copy(), left_cy.copy(), sub_width))
            segments.append((right_cx.copy(), right_cy.copy(), sub_width))
            bars.append((left_cx.copy(), left_cy.copy(),
                         right_cx.copy(), right_cy.copy()))

            prev_idx = conf_idx
            next_spacing = np.random.uniform(2, 4) * self.thread_width
            current_s = conf_s + next_spacing

        # Final trunk segment
        if prev_idx < n - 10:
            segments.append((
                trunk_cx[prev_idx:].copy(),
                trunk_cy[prev_idx:].copy(),
                self.thread_width,
            ))

        return segments, bars

    # ------------------------------------------------------------------
    # Segment placement
    # ------------------------------------------------------------------

    def _place_segment(self, cx, cy, width, chelev, iteration_id,
                       total_iterations):
        """Carve a single channel segment into the 3D grid."""
        in_grid = ((cx > self.xmin) & (cx < self.xmax) &
                   (cy > self.ymin) & (cy < self.ymax))
        if in_grid.sum() < 8:
            return
        first = int(np.argmax(in_grid))
        last = len(in_grid) - 1 - int(np.argmax(in_grid[::-1]))
        cx = cx[first:last + 1]
        cy = cy[first:last + 1]

        result = self._compute_curvature(cx, cy, width=width)
        if result[0] is None:
            return
        cx, cy, vx, vy, curv, thalweg, chwidth, length = result

        if cx.size < 4:
            return

        genchannel(
            width, self.xsiz, self.ysiz,
            chelev, self.zsiz,
            self.nx, self.ny, self.nz,
            cx, cy, self.x, self.y,
            vx, vy, curv,
            0.9, 5 * self.zsiz, 1, 0, 0,
            iteration_id, self.facies, self.poro, self.poro0,
            thalweg, chwidth, self.dwratio,
            [10000000000], total_iterations,
        )

    # ------------------------------------------------------------------
    # Grid-based bar filling (replaces per-BBC bar filling)
    # ------------------------------------------------------------------

    def _fill_bars_grid(self, chelev):
        """Fill bar facies as a halo around channel threads.

        Uses morphological dilation to expand the channel mask by a
        limited radius (~1 thread-width), then fills only the expanded
        (non-channel) cells with bar facies.  This produces bar islands
        that are proportional to the channels instead of dominating the
        plan view.
        """
        idz = int(chelev / self.zsiz)
        if idz >= self.nz:
            return

        center_y = (self.ymin + self.ymax) / 2.0
        half_w = self.braidplain_width / 2.0

        # Boolean mask for y-indices within braidplain
        bp_mask = ((self.y >= center_y - half_w) &
                   (self.y <= center_y + half_w))

        bar_depth = max(2, int(np.ceil(
            self.dwratio * self.thread_width / self.zsiz)))
        z_top = min(idz, self.nz)
        z_bot = max(0, idz - bar_depth)
        if z_bot >= z_top:
            return

        bar_poro = self.bar_poro_factor * self.poro0

        # Dilation radius: expand channels by ~1 thread width in cells
        cell_size = max(self.xsiz, self.ysiz)
        bar_radius = max(2, int(np.ceil(
            self.thread_width / cell_size)))

        # Disk-shaped structuring element
        yy, xx = np.ogrid[-bar_radius:bar_radius + 1,
                          -bar_radius:bar_radius + 1]
        disk = (xx * xx + yy * yy) <= bar_radius * bar_radius

        # 2D braidplain mask for restricting dilation
        bp_mask_2d = np.zeros((self.nx, self.ny), dtype=bool)
        bp_mask_2d[:, bp_mask] = True

        for iz in range(z_bot, z_top):
            facies_z = self.facies[:, :, iz]
            poro_z = self.poro[:, :, iz]

            # Channel mask within braidplain (channels are facies=1
            # during simulation, before the final remap)
            channel_mask = np.zeros((self.nx, self.ny), dtype=bool)
            channel_mask[:, bp_mask] = (facies_z[:, bp_mask] == 1)

            if not channel_mask.any():
                continue

            # Dilate channel mask by disk radius
            expanded = binary_dilation(channel_mask, structure=disk)
            expanded &= bp_mask_2d

            # Fill only newly expanded cells (not already channel/bar)
            fill_mask = expanded & (~channel_mask) & (facies_z == 0)
            facies_z[fill_mask] = 2   # bar (before remap)
            poro_z[fill_mask] = bar_poro

    # ------------------------------------------------------------------
    # BBC bar filling (lens between sub-threads)
    # ------------------------------------------------------------------

    def _fill_bar_between(self, left_cx, left_cy, right_cx, right_cy,
                          chelev):
        """Fill bar facies in the lens between two sub-threads."""
        idz = int(chelev / self.zsiz)
        if idz >= self.nz:
            return

        bar_depth = max(2, int(np.ceil(
            self.dwratio * self.thread_width / self.zsiz)))
        z_top = min(idz, self.nz)
        z_bot = max(0, idz - bar_depth)
        if z_bot >= z_top:
            return

        # Subsample scan-lines
        n_pts = left_cx.size
        step = max(1, n_pts // 200)
        l_cx = left_cx[::step]
        l_cy = left_cy[::step]
        r_cx = right_cx[::step]
        r_cy = right_cy[::step]

        max_sep = np.sqrt((l_cx - r_cx)**2 + (l_cy - r_cy)**2).max()
        n_fill = max(5, int(2 * max_sep / min(self.xsiz, self.ysiz)))

        fracs = np.linspace(0, 1, n_fill).reshape(1, -1)
        fill_x = l_cx.reshape(-1, 1) + fracs * (r_cx - l_cx).reshape(-1, 1)
        fill_y = l_cy.reshape(-1, 1) + fracs * (r_cy - l_cy).reshape(-1, 1)

        ix = ((fill_x - self.xmin) / self.xsiz).astype(int).ravel()
        iy = ((fill_y - self.ymin) / self.ysiz).astype(int).ravel()

        valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        ix = ix[valid]
        iy = iy[valid]

        bar_poro = self.bar_poro_factor * self.poro0
        for iz in range(z_bot, z_top):
            bg_mask = self.facies[ix, iy, iz] == 0
            self.facies[ix[bg_mask], iy[bg_mask], iz] = 2
            self.poro[ix[bg_mask], iy[bg_mask], iz] = bar_poro

    # ------------------------------------------------------------------
    # Main simulation
    # ------------------------------------------------------------------

    def simulation(self, n_channels=10):
        """Run braided channel simulation.

        For each depositional event, generates n_threads channel paths
        at evenly-spaced y-positions within the braidplain.  Each path
        may bifurcate into sub-threads around mid-channel bars.

        The same channel pattern is stamped across a vertical band of
        z-levels before the next event generates a new pattern.  This
        ensures smooth vertical continuity (the same channel geometry
        appears at consecutive z-levels).
        """
        min_chelev = (self.cz + 1) * self.zsiz
        max_chelev = (self.nz - 1) * self.zsiz
        center_y = (self.ymin + self.ymax) / 2.0
        half_width = self.braidplain_width / 2.0

        # Channel depth in model units
        channel_depth = max(self.dwratio * self.thread_width,
                            2 * self.zsiz)

        # Each event covers a z-band; stamp the same pattern at
        # multiple chelev values within that band for continuity.
        z_range = max_chelev - min_chelev
        z_band = z_range / max(n_channels, 1)
        # Step between passes: 50% of channel_depth for generous overlap
        z_step = max(self.zsiz, channel_depth * 0.5)
        passes_per_event = max(1, int(np.ceil(z_band / z_step)))

        seg_counter = 0
        total_segs_est = (n_channels * passes_per_event
                          * self.n_threads * 6)

        # Lane width for evenly-spacing threads
        lane_width = self.braidplain_width / self.n_threads

        for event in range(n_channels):
            event_base = min_chelev + event * z_band

            # ----------------------------------------------------------
            # Generate thread patterns ONCE for this depositional event
            # ----------------------------------------------------------
            thread_data = []  # list of (segments, bbc_bars)

            for t_idx in range(self.n_threads):
                base_y = (center_y - half_width
                          + (t_idx + 0.5) * lane_width)
                jitter = np.random.uniform(-0.3 * lane_width,
                                           0.3 * lane_width)
                y0 = base_y + jitter

                trunk_cx, trunk_cy = self._generate_thread_streamline(
                    y0=y0, s=0.08)
                if trunk_cx is None:
                    continue

                trunk_cy = self._constrain_to_braidplain(
                    trunk_cy, center_y)

                # Contiguous in-grid slice
                in_grid = (
                    (trunk_cx > self.xmin) & (trunk_cx < self.xmax) &
                    (trunk_cy > self.ymin) & (trunk_cy < self.ymax))
                if in_grid.sum() < 20:
                    continue
                first = int(np.argmax(in_grid))
                last = (len(in_grid) - 1
                        - int(np.argmax(in_grid[::-1])))
                trunk_cx = trunk_cx[first:last + 1]
                trunk_cy = trunk_cy[first:last + 1]

                segments, bbc_bars = self._build_braided_path(
                    trunk_cx, trunk_cy)
                thread_data.append((segments, bbc_bars))

            if not thread_data:
                continue

            # ----------------------------------------------------------
            # Stamp this pattern across the z-band
            # ----------------------------------------------------------
            for z_pass in range(passes_per_event):
                chelev = event_base + z_pass * z_step
                if chelev / self.zsiz >= self.nz:
                    break

                for segments, bbc_bars in thread_data:
                    for seg_cx, seg_cy, seg_width in segments:
                        seg_counter += 1
                        self._place_segment(
                            seg_cx, seg_cy, seg_width, chelev,
                            iteration_id=seg_counter,
                            total_iterations=total_segs_est,
                        )
                    for bar in bbc_bars:
                        self._fill_bar_between(*bar, chelev)

                self._fill_bars_grid(chelev)

        # Remap facies so code order matches reservoir quality:
        # During simulation: genchannel() wrote channels as 1, bars as 2.
        # Final codes: 0=shale, 1=bar (intermediate), 2=channel (best).
        tmp = self.facies.copy()
        self.facies[tmp == 1] = 2  # channel → 2 (best)
        self.facies[tmp == 2] = 1  # bar → 1 (intermediate)
