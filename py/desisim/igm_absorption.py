"""
desisim.igm_absorption
========================

Task #35: reconciles this fork's pre-existing Lyman-alpha forest +
damped-Lyman-alpha (DLA) intervening-absorption machinery with the
additive-ground-truth-decomposition framework the rest of this project
uses.

--------------------------------------------------------------------------
Why this exists
--------------------------------------------------------------------------
Before this task, Lyman-alpha forest / DLA / metal-line intervening
absorption existed in THREE separate legacy code paths, none additive and
none reachable from this fork's own orchestrator (mock_spectrum.py):

  1. templates.py's legacy QSO.make_templates() already calls a self-
     contained forest generator (self.lyamock_maker, a lya_mock_p1d.
     MockMaker instance) and multiplies the result directly onto flux.
  2. SIMQSO.make_templates() has its own internal lyaforest flag that
     toggles simqso's own built-in forest noise model -- also
     multiplicative, internal to simqso's continuum construction.
  3. lya_spectra.get_spectra() reads pre-computed transmission skewers
     from an EXTERNAL CoLoRe-format FITS file and multiplies them onto
     flux, then optionally calls dla.py's insert_dlas() and multiplies
     that in too.

None of these three is reachable from generate_qso_component() -- so
every QSO mock built through this project's own pipeline had zero IGM
absorption physics, a large omission for anything at z gtrsim 2 (the
whole region blueward of rest-frame Lyalpha, 1215.67A, is dominated by
forest absorption in a real spectrum).

--------------------------------------------------------------------------
Which forest engine: MockMaker, not the external CoLoRe file
--------------------------------------------------------------------------
lya_mock_p1d.MockMaker generates a lognormal Gaussian random field
matching the McDonald et al. (2006) empirically-measured 1D Lyalpha
forest power spectrum P_1D(k,z), transformed to optical depth via a
redshift-evolving mean-opacity calibration -- genuinely self-contained,
no external file needed, already proven inside legacy
QSO.make_templates(). Real external CoLoRe/"London mock" files exist
specifically to preserve *correlated* large-scale structure BETWEEN
different QSO sightlines on the sky (important for BAO/clustering
science) -- but since this project's NPE trains on one spectrum at a
time, that cross-sightline correlation is not needed here. A single-
sightline stochastic generator that gets the 1D power spectrum right
is sufficient for this project's purposes and requires no external data
product, matching every other channel's fully self-contained, seed-
driven convention. Per PI direction, MockMaker is therefore the engine
used here; the external-file path is NOT wired in (a future opt-in
alternative is possible but out of scope).

--------------------------------------------------------------------------
DLA: dla.py reused as-is
--------------------------------------------------------------------------
dla.py's insert_dlas()/dla_spec() (Prochaska et al. 2008 dN/dz incidence,
an NHI column-density distribution spline, and an exact Voigt-profile
optical depth via scipy.special.wofz) is already a good, self-contained,
forest-source-agnostic physical model -- it only needs an observed-frame
wavelength array and the QSO's redshift, so it layers on top of
MockMaker's skewer unchanged.

--------------------------------------------------------------------------
Metal-line absorption: deferred
--------------------------------------------------------------------------
lya_spectra.py's apply_metals_transmission() does not independently model
metal lines -- it scales a fixed coefficient off the Lyalpha forest's own
optical depth as a crude proxy, already flagged as approximate in its own
docstring. Per PI direction this is deferred, not part of this task.

--------------------------------------------------------------------------
Not rest-frame-intrinsic: zqso is a required, non-drawn input
--------------------------------------------------------------------------
Every other module wired into mock_spectrum.py (Fe II, Balmer, dust,
broad lines, ...) is a REST-FRAME-intrinsic property of the source --
mock_spectrum.py is explicitly documented as "rest-frame only, no
redshifting." IGM absorption breaks that pattern unavoidably: it is
physically the transmission of intervening gas along the actual path
from the QSO to us, so it depends on the QSO's actual redshift, not just
its rest-frame spectral shape. This module resolves that by taking zqso
as a required (not drawn, not optional-with-a-prior) argument, computing
the transmission in observed-frame internally, and returning the result
re-indexed onto REST-frame wavelength (wave_obs = wave_rest*(1+zqso)) --
a valid, well-defined rest-frame curve for this ONE fixed, known
redshift, even though the underlying physics is not itself a rest-frame-
intrinsic property of the source. generate_qso_component() treats zqso
as an opt-in parameter (default None -> this module is skipped entirely,
exact previous behavior preserved) rather than drawing/requiring it,
since redshift assignment is the broader pipeline's job, not this
module's.

--------------------------------------------------------------------------
Multiplicative -> additive: same pattern as dust.py
--------------------------------------------------------------------------
Transmission is multiplicative by definition (F_obs = F_intrinsic * T),
so the additive deficit is EXACT, not an approximation:

    igm_flux(wave) = flux_to_absorb(wave) * (T_forest(wave)*T_DLA(wave) - 1)

This mirrors DustAttenuation.spectrum()'s own (wave, flux_to_attenuate,
...) -> additive-deficit calling convention exactly (see dust.py) --
spectrum() below takes the flux to absorb directly as an argument and
returns the already-computed additive deficit, not a transmission curve
the caller has to apply themselves. Per PI-approved design, the
flux_to_absorb passed by generate_qso_component() is the FULL pre-IGM QSO
flux (continuum_agn + broad_emission + feii_flux + balmer_flux), not just
continuum_agn alone (unlike dust/ism/associated, which this fork already
computes against continuum_agn alone as an established simplification) --
real Lyalpha forest famously eats into the blue wing of a QSO's own
Lyalpha emission line and other blue emission features too, so using only
the continuum here would miss a real, visible effect this task exists to
capture. Following the same "each deficit channel computed independently
against a shared baseline, summed linearly" convention already used for
ism_absorption/associated_absorption_flux/dust_flux (see decompose.py),
igm_flux lands in the "absorption" bucket alongside them.
"""

import numpy as np

from desisim.lya_mock_p1d import MockMaker
from desisim.dla import insert_dlas

LAMBDA_LYA = 1215.67  # rest-frame vacuum wavelength [A] -- matches lya_spectra.py's lambda_RF_LYA


class IGMAbsorption(object):
    """Additive Lyman-alpha-forest + DLA intervening-absorption channel.
    See module docstring for the full physical/design rationale.
    """

    def __init__(self, minwave=900.0, maxwave=1300.0, cdelt_kms=20.0, log10wave=None,
                 mockmaker_N2=15, mockmaker_dv_kms=10.0):
        """
        Args:
            minwave, maxwave (float): rest-frame output grid bounds [A].
                Only used if log10wave is not provided. Defaults bracket
                the rest-frame Lyalpha region (1215.67A); IGM absorption
                is a no-op (T=1) everywhere redward of the QSO's own
                Lyalpha, so a wider maxwave is harmless but only the
                blueward portion ever shows any effect.
            cdelt_kms (float): output-grid pixel size [km/s] (log-uniform
                grid, same convention as every other module in this fork).
            log10wave (ndarray, optional): explicit output log10(wave)
                grid, to match an external continuum/emission grid (this
                is how mock_spectrum.py always constructs it in practice).
            mockmaker_N2, mockmaker_dv_kms: forwarded to lya_mock_p1d.
                MockMaker's own constructor (skewer length 2**N2 cells of
                width mockmaker_dv_kms km/s each) -- structural parameters
                of the underlying forest generator, not astrophysical
                priors; the defaults match legacy QSO.make_templates()'s
                own MockMaker() call exactly.
        """
        from desisim.templates import C_LIGHT
        if log10wave is None:
            cdelt_loglam = cdelt_kms / C_LIGHT / np.log(10)
            log10wave = np.arange(np.log10(minwave), np.log10(maxwave), cdelt_loglam)
        self.log10wave = log10wave
        self._mockmaker_N2 = mockmaker_N2
        self._mockmaker_dv_kms = mockmaker_dv_kms

    def transmission(self, zqso, add_dlas=True, seed=None):
        """Combined forest*DLA multiplicative transmission.

        Args:
            zqso (float): QSO redshift. Required -- see module docstring
                ("Not rest-frame-intrinsic").
            add_dlas (bool): whether to inject DLAs via dla.py's
                insert_dlas() on top of the forest skewer (default True --
                this is real, calibrated incidence physics, not a
                placeholder, so it is on by default; pass False to
                disable).
            seed (int, optional): RNG seed for reproducibility.

        Returns:
            Tuple of (T, wave, params): T is the transmission array
            [npix] in [0, 1], indexed on THIS object's rest-frame wave
            grid but only physically valid for the given zqso (see module
            docstring); wave is 10**self.log10wave; params is a dict with
            zqso, add_dlas, and dla_meta (an astropy Table of injected
            DLAs' NHI/z, or None if add_dlas=False or none were drawn).
        """
        rand = np.random.RandomState(seed)
        wave_rest = 10 ** self.log10wave
        wave_obs = wave_rest * (1.0 + zqso)

        mm = MockMaker(N2=self._mockmaker_N2, dv_kms=self._mockmaker_dv_kms,
                        seed=int(rand.randint(0, 2**31 - 1)))
        skewer_wave, skewer_flux_batch = mm.get_lya_skewers(Ns=1)
        skewer_flux = skewer_flux_batch[0].copy()
        # No forest redward of the QSO's own Lyalpha emission, by
        # definition (matches legacy QSO.make_templates()'s own masking).
        skewer_flux[skewer_wave > LAMBDA_LYA * (1.0 + zqso)] = 1.0
        # left=1.0/right=1.0: outside the skewer's own coverage there is
        # no data: a conservative (no-absorption) fallback, not a
        # physical claim -- see minwave/maxwave's docstring note on
        # choosing a wave range that overlaps the skewer's actual extent.
        forest_T = np.interp(wave_obs, skewer_wave, skewer_flux, left=1.0, right=1.0)

        dla_meta = None
        if add_dlas:
            dlas, dla_model = insert_dlas(wave_obs, zqso, seed=int(rand.randint(0, 2**31 - 1)))
            dla_model = np.asarray(dla_model)
            if dla_model.shape != wave_obs.shape:
                # Degenerate edge case (see dla.py's insert_dlas: an
                # unusably narrow forest window returns dlas=[],
                # dla_model=[] instead of an all-ones array) -- treat as
                # no-DLA rather than propagate a shape mismatch.
                dla_model = np.ones_like(wave_obs)
            elif len(dlas) > 0:
                from astropy.table import Table
                dla_meta = Table()
                dla_meta['NHI'] = [d['N'] for d in dlas]
                dla_meta['z'] = [d['z'] for d in dlas]
        else:
            dla_model = np.ones_like(wave_obs)

        T_total = forest_T * dla_model
        params = dict(zqso=zqso, add_dlas=add_dlas, dla_meta=dla_meta,
                      ndla=0 if dla_meta is None else len(dla_meta))
        return T_total, wave_rest, params

    def spectrum(self, flux_to_absorb, zqso, add_dlas=True, seed=None):
        """Additive IGM-absorption deficit flux.

        Args:
            flux_to_absorb (ndarray): the flux array [npix, same grid as
                this object's wave] that IGM absorption acts on -- per PI-
                approved design this is the full pre-IGM QSO flux
                (continuum_agn + broad_emission + feii_flux + balmer_flux),
                not continuum_agn alone (see module docstring).
            zqso, add_dlas, seed: see transmission().

        Returns:
            Tuple of (igm_flux, wave, params): igm_flux is the additive
            deficit flux_to_absorb*(T-1) [npix], always <= 0 (same
            convention as ism_absorption/associated_absorption_flux/
            dust_flux); wave and params as returned by transmission().
        """
        T, wave, params = self.transmission(zqso, add_dlas=add_dlas, seed=seed)
        flux_to_absorb = np.asarray(flux_to_absorb)
        igm_flux = flux_to_absorb * (T - 1.0)
        return igm_flux, wave, params
