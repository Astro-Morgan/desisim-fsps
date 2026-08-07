"""
desisim.associated_absorption
==============================

Stochastic, multi-system narrow associated/intrinsic absorption-line
ground truth for QSO spectra: an independently-drawn NUMBER of discrete
absorbing systems, each at its own blueshifted velocity offset relative
to the QSO systemic redshift and its own velocity dispersion, each
imprinting a common set of UV resonance transitions (Mg II, C IV, Si IV,
O VI, C III], ...) at that system's shared kinematics.

--------------------------------------------------------------------------
Why this exists / what it replaces
--------------------------------------------------------------------------
This corrects a prior misreading of the PI's original request. What was
tracked as two separate items -- "parametric/stochastic BAL model" (a
from-scratch broad-absorption-trough model) and "two-component birth-
cloud/diffuse-ISM dust" -- was NOT what was meant. The actual request:
quasars frequently show MULTIPLE, DISCRETE, NARROW (not broad-trough)
absorption-line systems blueward of systemic, primarily in Mg II but also
C IV, C III], and other UV resonance transitions (Ne, Si, ...), each
system having its OWN velocity offset and its OWN velocity dispersion --
e.g. three separate Mg II absorption systems at three different velocity
offsets in a single QSO spectrum, each with a different linewidth. This
module implements exactly that, replacing both prior (incorrect) tasks.

This is a genuinely different physical/statistical object from
absorption.py's AbsorptionSpectrum: that module places a SINGLE draw of
each of 6 fixed ISM/CGM lines at ONE shared galaxy-frame kinematic
(zshift, one sigma_kms for the whole population) -- appropriate for
diffuse host-galaxy/CGM gas along the line of sight. This module instead
draws a RANDOM NUMBER of discrete, kinematically-INDEPENDENT absorbing
clouds/systems (each with its own velocity offset AND its own velocity
dispersion), each of which activates a whole set of transitions at once --
appropriate for the population of individually-resolved outflow/associated
absorption systems seen in real QSO spectra (Weymann et al. 1979;
Foltz et al. 1986; Hamann & Sabra 2004; Vestergaard 2003; and many
individual-system studies of associated Mg II/C IV/Si IV absorption).
It is also NOT bal.py's whole-real-template BAL mechanism (that remains
untouched, covering the separate broad-trough phenomenon); this module's
systems are narrow (per-system sigma_kms drawn from a Maxwell-Boltzmann
distribution -- see below -- typically tens to a couple hundred km/s, not
BAL's several-thousand-km/s-wide troughs).

--------------------------------------------------------------------------
Statistical model
--------------------------------------------------------------------------
Per the PI's explicit specification: "Poisson distributed (poisson in
blueward velocity offset, poisson in count, MB in velocity dispersion per
system)". Read literally, these three ingredients are exactly the three
defining properties of a single, standard object: a homogeneous POISSON
POINT PROCESS along the blueward velocity axis, plus an independent
Maxwell-Boltzmann draw per resulting point for that system's own velocity
dispersion. Implementing this as one coherent process (rather than
drawing "count" and "offsets" as two separately-motivated random
mechanisms) is both mathematically correct and avoids double-modeling,
since a homogeneous Poisson process's count and point-placement
statistics are not independent choices -- they are two properties of the
SAME underlying process, related by a standard, exact identity:

    For a homogeneous Poisson process with rate R [systems / (km/s)] on
    the blueward interval [0, V_MAX_KMS]:
      (1) The number of points N in the interval is itself Poisson
          distributed: N ~ Poisson(R * V_MAX_KMS).
      (2) CONDITIONAL ON N, the N point locations {t_1, ..., t_N} are
          independently and identically distributed as
          Uniform(0, V_MAX_KMS) (order statistics of N uniform draws).

    This is a standard, EXACT (not approximate) result from point-process
    theory (e.g. Kingman 1993, "Poisson Processes", Ch. 2): a homogeneous
    Poisson process's defining property is independent, uniformly-
    distributed increments conditional on the count in any interval,
    which is precisely statement (2). It provides an exact algorithm for
    simulating the process (rather than the equivalent but more awkward
    approach of drawing cumulative Exponential(1/R) inter-arrival
    spacings): draw N ~ Poisson(R*V_MAX_KMS), then draw N iid
    Uniform(0, V_MAX_KMS) values and negate them (to place them BLUEWARD
    of systemic, i.e. v_i = -t_i <= 0).

This module parametrizes the process by its MEAN COUNT over the full
window (MEAN_N_SYSTEMS = R * V_MAX_KMS) rather than the rate R directly,
since the mean count is the more physically intuitive knob; R is
recovered internally as MEAN_N_SYSTEMS / V_MAX_KMS.

Each system's own velocity dispersion sigma_kms_i is then drawn
independently from a Maxwell-Boltzmann distribution (scipy.stats.maxwell,
scale parameter MB_SCALE_KMS), exactly as specified ("MB in velocity
dispersion per system"). The Maxwell-Boltzmann shape (positive-definite,
unimodal, right-skewed) is the standard choice in the absorption-line
literature for ensembles of cloud/component velocity widths (directly
analogous to the "b-distribution" long used for Lyman-alpha-forest
Doppler parameters, e.g. Hui & Rutledge 1999), and is a genuinely
different, richer shape than the log-uniform sigma_kms draw
AbsorptionSpectrum uses for its single shared ISM/CGM kinematic scale.

--------------------------------------------------------------------------
Per-system line list
--------------------------------------------------------------------------
Each system activates the SAME shared kinematics (velocity offset,
velocity dispersion) across whichever transitions are active, but each
transition gets its OWN independently-drawn peak optical depth tau0
(reflecting genuinely different column density/oscillator-strength/
ionization-fraction combinations for different ions from the same physical
cloud) -- identical convention to AbsorptionSpectrum's per-line tau0 draws.

Default transitions (LINE_WAVE_VACUUM below), per the PI's explicit list
("primarily MgII, but also CIII] and CIV and Ne and Si and Ar etc"):
2026-08-07 UPDATE: the PI provided the exact, authoritative source this
list should match -- the desihub/prospect viewer's own
absorption_lines.txt/emission_lines.txt data files (the same line
definitions used by the legacysurvey.org DESI spectrum viewer). The
default transition list below is now built directly from those files
(every entry with vacuum wavelength < Mg II's 2796.35A), replacing the
earlier ad hoc guesses (a previous version of this module included
speculative Ne VIII/Ar I doublets as placeholder interpretations of the
PI's underspecified "Ne"/"Ar" -- both are DROPPED now that the
authoritative list is available and does not include either ion).

From prospect's absorption_lines.txt (curated absorption transitions,
included verbatim -- these are real, standard interstellar/CGM/associated
absorption features, e.g. the classic Fe II UV1/UV2/UV3-multiplet lines
routinely seen alongside Mg II absorbers):
    Si II 1260.4221, O I 1302.1685, Si II 1304.3702, C II 1334.5323,
    Si IV 1393.75/1402.77, S V 1501.76, Si II 1526.7070, C IV 1549.48,
    Fe II 1608.4509, Al II 1670.788, Si II 1808.0129,
    Al III 1854.7183/1862.7911, Mg I 2026.4768,
    Fe II 2344.2130/2374.4603/2382.7642/2389.2396,
    Mg II 2796.3543/2803.5315 (the doublet itself, kept as the reference
    transition around which "blueward" is defined).

From prospect's emission_lines.txt, only the subset that are genuine
PERMITTED/RESONANCE transitions (i.e. plausible in absorption -- connect
to the ground state with a real oscillator strength) is added; the
file's forbidden/semi-forbidden/fluorescent/recombination entries
([Ne IV], C II], O III], Si III], He II 1640, and the excited-state
"Si II*" fluorescent lines) are excluded as not being physically sensible
ground-state absorbers:
    Ly-beta 1025.18, Ly-alpha 1215.67, N V (true doublet
    1238.821/1242.804, used here in place of the viewer's single blended
    "N V 1240" marker for consistency with how every other doublet in
    this module is treated), C III] 1908.734 (semi-forbidden; RETAINED
    per explicit PI direction from the original request despite being
    atypical for absorption -- flagged as before).

One addition beyond the viewer's own list, per the PI's explicit "plus
any others its missing": O VI 1031.93/1037.62 (doublet). This is the
single most standard, well-precedented high-ionization associated/
intrinsic-absorption doublet in the QSO literature (routinely discussed
alongside N V/C IV in mini-BAL and associated-absorber studies) that the
viewer's own list happens not to include -- a well-justified, not
speculative, addition (unlike the earlier dropped Ne VIII/Ar I guesses).

All wavelengths above are exactly as provided in the PI's
absorption_lines.txt/emission_lines.txt excerpt (2026-08-07), except
O VI (standard NIST ASD vacuum values) and the N V doublet split
(standard NIST ASD components of the viewer's blended 1240.81 marker).

--------------------------------------------------------------------------
Additive-flux-deficit convention (same pattern as absorption.py)
--------------------------------------------------------------------------
Exactly AbsorptionSpectrum's convention: optical depths from independent
systems simply add (standard radiative transfer for a superposition of
absorbers along one sightline) before the single exp(-tau_total) is
applied, and the module returns the additive flux DEFICIT relative to the
supplied continuum:

    tau_total(lambda) = sum_{i=1}^{N} tau_system_i(lambda)
    absorption_flux(lambda) = continuum(lambda) * (exp(-tau_total(lambda)) - 1)

which is <=0 everywhere by construction, applies only to the continuum
(same explicit scope choice as AbsorptionSpectrum -- see that module's
docstring for the rationale), and is meant to be summed alongside
AbsorptionSpectrum's own ISM/CGM output and DustAttenuation's output in
decompose.py's "absorption" bucket.

--------------------------------------------------------------------------
Scope
--------------------------------------------------------------------------
Independently testable, standalone module -- not wired into
QSO.make_templates() here (same convention as every other new component
in this fork). N=0 systems (a real, non-trivial-probability outcome of
the Poisson draw) is handled explicitly and returns an all-zero deficit.
Every tau0/rate/scale parameter here is an independent free parameter
with a MAGIC-flagged (or, for f_scat-style empirically-grounded cases,
explicitly cited) prior, pending this project's planned NPE/normalizing-
flow calibration against real spectra -- same convention as every other
new component in this fork.
"""

import numpy as np

from desisim.templates import _use_torch_backend, _lines_to_spectrum_numpy, _lines_to_spectrum_torch, C_LIGHT


class AssociatedAbsorberSystems(object):
    """Stochastic, multi-system narrow associated-absorption ground truth
    for QSO spectra. See module docstring for the full statistical model
    (Poisson-process system count/placement, Maxwell-Boltzmann per-system
    velocity dispersion) and the per-transition line list.
    """

    TRANSITION_NAMES = [
        # -- prospect absorption_lines.txt, verbatim (< Mg II 2796.35A) --
        # C IV is split into its true 1548.204/1550.781 doublet (NIST ASD)
        # rather than prospect's single blended "C IV 1549.48" marker, for
        # consistency with how every other doublet in this module is
        # treated (Mg II, Si IV, N V, O VI all use real doublet pairs).
        'SiII_1260', 'OI_1302', 'SiII_1304', 'CII_1334', 'SiIV_1393', 'SiIV_1402',
        'SV_1501', 'SiII_1526', 'CIV_1548', 'CIV_1550', 'FeII_1608', 'AlII_1670',
        'SiII_1808', 'AlIII_1854', 'AlIII_1862', 'MgI_2026', 'FeII_2344', 'FeII_2374',
        'FeII_2382', 'FeII_2389', 'MgII_2796', 'MgII_2803',
        # -- prospect emission_lines.txt, resonance/permitted subset only --
        'Lyb_1025', 'Lya_1215', 'NV_1238', 'NV_1242', 'CIII]_1908',
        # -- PI-directed addition beyond the viewer's list --
        'OVI_1031', 'OVI_1037',
    ]

    # Vacuum wavelengths [Angstrom]. See module docstring's "Per-system
    # line list" for exact sourcing (desihub/prospect's
    # absorption_lines.txt/emission_lines.txt, provided verbatim by the
    # PI 2026-08-07, plus the O VI addition and N V doublet split).
    LINE_WAVE_VACUUM = {
        'SiII_1260':  1260.4221,
        'OI_1302':    1302.1685,
        'SiII_1304':  1304.3702,
        'CII_1334':   1334.5323,
        'SiIV_1393':  1393.75,
        'SiIV_1402':  1402.77,
        'SV_1501':    1501.76,
        'SiII_1526':  1526.7070,
        'CIV_1548':   1548.204,
        'CIV_1550':   1550.781,
        'FeII_1608':  1608.4509,
        'AlII_1670':  1670.788,
        'SiII_1808':  1808.0129,
        'AlIII_1854': 1854.7183,
        'AlIII_1862': 1862.7911,
        'MgI_2026':   2026.4768,
        'FeII_2344':  2344.2130,
        'FeII_2374':  2374.4603,
        'FeII_2382':  2382.7642,
        'FeII_2389':  2389.2396,
        'MgII_2796':  2796.3543,
        'MgII_2803':  2803.5315,
        'Lyb_1025':   1025.18,
        'Lya_1215':   1215.67,
        'NV_1238':    1238.821,
        'NV_1242':    1242.804,
        'CIII]_1908': 1908.734,
        'OVI_1031':   1031.93,
        'OVI_1037':   1037.62,
    }

    # (Y) MAGIC: mean number of associated absorption systems per QSO
    # sightline (over the full blueward window V_MAX_KMS). Poisson-family
    # count statistics for QSO absorber populations are themselves
    # well-established in the literature (e.g. clustering statistics of
    # intervening/associated Mg II absorbers), but this specific mean
    # value is a placeholder, not a fit to any dataset.
    MEAN_N_SYSTEMS = 1.5

    # (Y) MAGIC: maximum blueward extent of the Poisson process [km/s].
    # Matches the same order-of-magnitude AGN-outflow velocity range
    # already used elsewhere in this fork's gap-analysis discussion
    # (CIV-blueshift-like offsets up to a few thousand km/s).
    V_MAX_KMS = 3000.0

    # (Y) MAGIC: Maxwell-Boltzmann scale parameter [km/s] for per-system
    # velocity dispersion (scipy.stats.maxwell's `scale`). Mean of a
    # Maxwell-Boltzmann distribution is scale*sqrt(8/pi) ~ 1.596*scale;
    # MB_SCALE_KMS=50 gives a mean sigma_kms ~ 80 km/s and a
    # positive-definite, right-skewed spread from a few 10s up to a few
    # hundred km/s -- consistent with typical individual-component
    # linewidths in the associated/mini-BAL absorption-system literature,
    # not a fit to any specific dataset.
    MB_SCALE_KMS = 50.0

    # (Y) MAGIC: independent log-normal priors on peak optical depth tau0
    # per transition, in log10(tau0) space -- same convention as
    # AbsorptionSpectrum.TAU0_PRIORS. Amplitude TIERS below are informed
    # by rough, literature-typical relative line strengths/oscillator
    # strengths (e.g. Fe II 2382 is the strongest of the four Fe II UV
    # lines here per its NIST f-value; Lyman-alpha is typically strong/
    # easily saturated even in weak systems; S V and Mg I trace rarer
    # ionization/neutral states and are set weaker) -- these are
    # order-of-magnitude placements, NOT a precise atomic-data fit, same
    # caveat as every other MAGIC prior in this fork. Doublet ratios
    # (bluer:redder ~2:1, i.e. redder = 0.5x bluer) follow the standard
    # alkali-like resonance-doublet oscillator-strength ratio already used
    # for Na I D/Mg II in absorption.py.
    TAU0_PRIORS = {
        # -- absorption_lines.txt transitions --
        'SiII_1260':  dict(mean=np.log10(0.25), sigma=0.6),
        'OI_1302':    dict(mean=np.log10(0.15), sigma=0.6),
        'SiII_1304':  dict(mean=np.log10(0.15), sigma=0.6),
        'CII_1334':   dict(mean=np.log10(0.2), sigma=0.6),
        'SiIV_1393':  dict(mean=np.log10(0.4), sigma=0.6),
        'SiIV_1402':  dict(mean=np.log10(0.4 * 0.5), sigma=0.6),
        'SV_1501':    dict(mean=np.log10(0.1), sigma=0.6),
        'SiII_1526':  dict(mean=np.log10(0.2), sigma=0.6),
        'CIV_1548':   dict(mean=np.log10(0.6), sigma=0.6),
        'CIV_1550':   dict(mean=np.log10(0.6 * 0.5), sigma=0.6),
        'FeII_1608':  dict(mean=np.log10(0.12), sigma=0.6),
        'AlII_1670':  dict(mean=np.log10(0.2), sigma=0.6),
        'SiII_1808':  dict(mean=np.log10(0.15), sigma=0.6),
        'AlIII_1854': dict(mean=np.log10(0.3), sigma=0.6),
        'AlIII_1862': dict(mean=np.log10(0.3 * 0.5), sigma=0.6),
        'MgI_2026':   dict(mean=np.log10(0.1), sigma=0.6),
        'FeII_2344':  dict(mean=np.log10(0.2), sigma=0.6),
        'FeII_2374':  dict(mean=np.log10(0.08), sigma=0.6),
        'FeII_2382':  dict(mean=np.log10(0.35), sigma=0.6),
        'FeII_2389':  dict(mean=np.log10(0.08), sigma=0.6),
        'MgII_2796':  dict(mean=np.log10(0.5), sigma=0.6),
        'MgII_2803':  dict(mean=np.log10(0.5 * 0.7), sigma=0.6),
        # -- emission_lines.txt resonance subset --
        'Lyb_1025':   dict(mean=np.log10(0.3), sigma=0.6),
        'Lya_1215':   dict(mean=np.log10(0.7), sigma=0.6),
        'NV_1238':    dict(mean=np.log10(0.4), sigma=0.6),
        'NV_1242':    dict(mean=np.log10(0.4 * 0.5), sigma=0.6),
        'CIII]_1908': dict(mean=np.log10(0.2), sigma=0.6),
        # -- PI-directed addition beyond the viewer's list --
        'OVI_1031':   dict(mean=np.log10(0.3), sigma=0.6),
        'OVI_1037':   dict(mean=np.log10(0.3 * 0.5), sigma=0.6),
    }

    def __init__(self, minwave=950.0, maxwave=3000.0, cdelt_kms=20.0,
                 log10wave=None, include_transitions=None):
        """
        Args:
            minwave, maxwave (float): rest-frame output grid bounds [A],
                wide enough by default to cover Ly-beta (1025.18A, the
                shortest default transition) through Mg II, with margin
                for the largest allowed blueward velocity offset
                (V_MAX_KMS). Only used if log10wave is not provided.
            cdelt_kms (float): output-grid pixel size [km/s] (log-uniform
                grid, same convention as AbsorptionSpectrum/EMSpectrum).
                Only used if log10wave is not provided.
            log10wave (ndarray, optional): explicit output log10(wave)
                grid, to match an external continuum grid exactly.
            include_transitions (list of str, optional): subset of
                TRANSITION_NAMES active in every system (default: all 29,
                matching the desihub/prospect viewer's own line list
                blueward of Mg II plus the O VI addition -- see module
                docstring). Every active system uses the same transition
                set.
        """
        if log10wave is None:
            cdelt_loglam = cdelt_kms / C_LIGHT / np.log(10)
            log10wave = np.arange(np.log10(minwave), np.log10(maxwave), cdelt_loglam)
        self.log10wave = log10wave

        if include_transitions is None:
            include_transitions = list(self.TRANSITION_NAMES)
        else:
            unknown = set(include_transitions) - set(self.TRANSITION_NAMES)
            if unknown:
                raise ValueError('Unknown transition name(s) in include_transitions: {}'.format(sorted(unknown)))
        self.include_transitions = include_transitions

    def spectrum(self, continuum_wave, continuum_flux, n_systems=None, velocities_kms=None,
                 sigma_kms=None, tau0=None, tau0_priors=None, zshift=0.0, seed=None,
                 backend='auto', device=None, dtype=None):
        """Build the additive multi-system associated-absorption flux
        deficit.

        Args:
            continuum_wave (ndarray): wavelength grid [A] of the continuum
                this absorption is applied to.
            continuum_flux (ndarray): continuum flux density on
                continuum_wave, same units as the desired output.
            n_systems (int, optional): explicit number of systems,
                overriding the Poisson draw (N ~
                Poisson(MEAN_N_SYSTEMS)).
            velocities_kms (ndarray, optional): explicit [n_systems]
                array of per-system blueward velocity offsets [km/s,
                <=0], overriding the Poisson-process placement (uniform
                on [0, V_MAX_KMS], negated). Length must equal n_systems
                if both are given explicitly.
            sigma_kms (ndarray, optional): explicit [n_systems] array of
                per-system velocity dispersions [km/s], overriding the
                Maxwell-Boltzmann draw.
            tau0 (list of dict, optional): explicit per-system peak
                optical depth overrides; tau0[i] is a dict keyed by
                transition name (subset of self.include_transitions) for
                system i. Unlisted (system, transition) pairs draw from
                tau0_priors.
            tau0_priors (dict, optional): override TAU0_PRIORS.
            zshift (float): QSO systemic redshift applied to all transition
                centers before the per-system blueward velocity offset
                (same convention as EMSpectrum/AbsorptionSpectrum's
                zshift); the output grid (self.log10wave) is unchanged.
            seed (int, optional): RNG seed for reproducibility.
            backend, device, dtype: see AbsorptionSpectrum.spectrum's
                docstring for the identical torch/numpy dispatch
                convention -- reuses the same _lines_to_spectrum_{numpy,
                torch} primitives, called once per system (each system
                has its own shared sigma_kms).

        Returns:
            Tuple of (absorption_flux, wave, systems), where
            absorption_flux is an array [npix] of flux-deficit values
            (<=0 everywhere, same units as continuum_flux); wave is
            10**self.log10wave; systems is a Table with one row per
            (system, transition) pair actually used (columns: system,
            transition, wave, velocity_kms, sigma_kms, tau0), empty
            (zero rows) if n_systems drew to 0.
        """
        from astropy.table import Table

        rand = np.random.RandomState(seed)

        if tau0_priors is None:
            tau0_priors = self.TAU0_PRIORS
        if tau0 is None:
            tau0 = []

        if n_systems is None:
            n_systems = rand.poisson(self.MEAN_N_SYSTEMS)

        if velocities_kms is None:
            # Exact Poisson-process placement: conditional on n_systems
            # points in [0, V_MAX_KMS], the locations are iid
            # Uniform(0, V_MAX_KMS) -- see module docstring's derivation.
            # Negated to represent BLUEWARD offsets (v <= 0).
            velocities_kms = -rand.uniform(0.0, self.V_MAX_KMS, size=n_systems)
        else:
            velocities_kms = np.asarray(velocities_kms, dtype=float)
            if velocities_kms.shape[0] != n_systems:
                raise ValueError('velocities_kms has {} entries, expected n_systems={}'.format(
                    velocities_kms.shape[0], n_systems))

        if sigma_kms is None:
            from scipy.stats import maxwell
            sigma_kms = maxwell.rvs(scale=self.MB_SCALE_KMS, size=n_systems,
                                     random_state=rand)
        else:
            sigma_kms = np.asarray(sigma_kms, dtype=float)
            if sigma_kms.shape[0] != n_systems:
                raise ValueError('sigma_kms has {} entries, expected n_systems={}'.format(
                    sigma_kms.shape[0], n_systems))

        names = list(self.include_transitions)
        wave_rest = np.array([self.LINE_WAVE_VACUUM[name] for name in names])

        wave_out = 10**self.log10wave
        tau_total = np.zeros_like(self.log10wave)

        table_rows = []
        for i in range(n_systems):
            system_tau0 = tau0[i] if i < len(tau0) else {}
            tau0_resolved = np.empty(len(names))
            for j, name in enumerate(names):
                if name in system_tau0:
                    tau0_resolved[j] = system_tau0[name]
                else:
                    prior = tau0_priors[name]
                    tau0_resolved[j] = 10**rand.normal(prior['mean'], prior['sigma'])

            log10sigma = sigma_kms[i] / C_LIGHT / np.log(10)
            # centers combine the QSO systemic redshift (zshift) with this
            # system's own blueward velocity offset (non-relativistic
            # Doppler approximation in log10-wavelength space, same
            # convention as sigma_kms's log10sigma conversion above and
            # elsewhere in this fork).
            centers = np.log10(wave_rest * (1.0 + zshift)) + velocities_kms[i] / C_LIGHT / np.log(10)

            in_window = (centers > self.log10wave.min()) & (centers < self.log10wave.max())

            if np.any(in_window):
                if _use_torch_backend(backend):
                    tau_system = _lines_to_spectrum_torch(self.log10wave, centers[in_window],
                                                           tau0_resolved[in_window], log10sigma,
                                                           device=device, dtype=dtype)
                else:
                    tau_system = _lines_to_spectrum_numpy(self.log10wave, centers[in_window],
                                                           tau0_resolved[in_window], log10sigma)
                tau_total = tau_total + tau_system

            for j, name in enumerate(names):
                table_rows.append((i, name, wave_rest[j], velocities_kms[i], sigma_kms[i],
                                    tau0_resolved[j], bool(in_window[j])))

        continuum_on_grid = np.interp(wave_out, continuum_wave, continuum_flux)
        absorption_flux = continuum_on_grid * (np.exp(-tau_total) - 1.0)

        systems = Table(rows=table_rows if table_rows else None,
                         names=('system', 'transition', 'wave', 'velocity_kms',
                                'sigma_kms', 'tau0', 'in_window'),
                         dtype=('i4', 'U16', 'f8', 'f8', 'f8', 'f8', 'bool'))

        return absorption_flux, wave_out, systems
