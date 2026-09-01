"""
desisim.fsps_continuum
========================

Generate a library of rest-frame stellar continuum spectra using
python-fsps, as an opt-in SUPPLEMENT to (not a replacement for)
desisim.io.read_basis_templates (handoff Sec 1.1).

Open Question 1 resolution: the handoff's own recommendation -- supplement
first, for easier A/B comparison, unless told otherwise -- is what's
implemented here. Nobody has said otherwise, so nothing about the default
read_basis_templates()-backed path changes.

Zero changes to GALAXY/ELG/BGS/LRG were needed for this: GALAXY.__init__
already accepts baseflux/basewave/basemeta overrides (pre-existing, not
added for this feature -- see py/desisim/templates.py's GALAXY.__init__
signature). Using an FSPS-generated continuum is therefore just:

    from desisim.fsps_continuum import fsps_basis_templates
    baseflux, basewave, basemeta = fsps_basis_templates(objtype='ELG', nbase=128, seed=1)
    elg = ELG(baseflux=baseflux, basewave=basewave, basemeta=basemeta)

Everything downstream (D4000-EW coupling, color cuts, magnitude
normalization, EMSpectrum) works identically regardless of which continuum
source produced (baseflux, basewave, basemeta), since GALAXY only ever
indexes into that triple positionally -- it has no idea where it came from.

python-fsps notes (verified directly against source/docs this session, see
SETUP.md Sec 4): FSPS has no continuum-only, pre-absorption stellar output
anywhere in its API -- add_neb_emission only controls the *nebular*
component. So the continuum generated here still contains real
photospheric absorption lines, by design, matching this project's
"stellar absorption is bundled with the continuum" decision (handoff Sec
0). Nebular emission stays desisim's own job (EMSpectrum);
add_neb_emission/add_neb_continuum are kept OFF below so nebular physics
isn't double-counted from two independent sources.

IMPORTANT CAVEATS discovered while implementing this (flagging rather than
silently shipping around them):

1. Premise check on "denser": the handoff's motivating assumption (Sec 0)
   was that $DESI_BASIS_TEMPLATES is a small (~5-eigentemplate) set. This
   session verified that assumption directly against the actual FITS files
   (not confirmed data, per the handoff's own caveat) and found it is
   false: the real ELG/BGS/LRG basis libraries have 7735/7636/3000
   individual real-galaxy-SED-fit templates respectively, not ~5. FSPS's
   value-add here is therefore NOT "more templates than a sparse basis" --
   it's (a) *continuous* metallicity interpolation (zcontinuous=1) instead
   of picking among a fixed, already-observed set, and (b) a
   forward-model with cleanly known, controllable generating parameters
   (age/Z/dust/SFH), vs. the real basis templates' parameters only being
   available as secondary SED-fit-derived quantities. That's still
   valuable for a mock generator whose whole point is having exact
   ground truth, but it's a different justification than "denser", and is
   worth knowing before deciding how much to invest here.
2. Resolution: the prebuilt PyPI python-fsps wheel used in this sandbox
   (see SETUP.md Sec 2 -- no Fortran compiler available here) ships with
   FSPS's spec_lib fixed at compile time to the "low-resolution" C3K
   library (confirmed via StellarPopulation.libraries == ('mist',
   'c3k_lr', 'DL07') and by inspecting sp.wavelengths directly: median
   spacing ~20 Angstrom near 4000A, i.e. R~200). That is *coarser* than
   the real DESI basis templates in this range (~0.7 Angstrom spacing,
   R~6000 near 4000A) -- the opposite of "denser" in the
   wavelength-sampling sense. Getting genuinely higher spectral resolution
   requires recompiling the FSPS Fortran backend from source with a
   different SPEC_LIB choice (e.g. the high-res c3k_hr variant, whose data
   files are already present under $SPS_HOME/SPECTRA/C3K/c3k_hr/ from the
   cconroy20/fsps clone) on a machine with a Fortran compiler (e.g.
   NERSC/Perlmutter) -- not possible in this sandbox. The code below does
   not hardcode any assumption about which spec_lib is compiled in; it
   reads sp.wavelengths directly, so recompiling FSPS elsewhere with
   c3k_hr would transparently produce a denser grid with no code changes.
3. Performance: sp.get_spectrum() cost in this environment ranged from
   ~0.7s to ~6.5s per call across different measurements in this session
   (regardless of which parameter changed -- logzsol, dust2, tau, even
   tage alone; this build does not appear to cache/incrementally update
   between calls under zcontinuous=1), i.e. noisy and clearly not
   dominated by which physics changed but by the sandbox's own variable
   load. nproc= support (multiprocessing.Pool, each worker gets its own
   StellarPopulation) is provided below and was verified for correctness
   (bit-identical output regardless of nproc for the same seed) and real
   (if sub-linear, ~1.4x on this sandbox's 2 cores) speedup -- but
   generating a "tens of thousands"-scale library was NOT completed in
   this session: the development sandbox has only 2 CPU cores and no
   ability to run a job across tool-call boundaries (each shell call is an
   isolated process; anything backgrounded dies when the call ends), so a
   multi-hour single-session job simply doesn't fit here. This is a
   correctness-and-scaling-mechanism deliverable, meant to be run at real
   scale on a many-core machine (e.g. a NERSC Perlmutter CPU node with 128
   cores) -- not a completed tens-of-thousands library.

--------------------------------------------------------------------------
Bursty SFH (task #37, added 2026-08-31)
--------------------------------------------------------------------------
Every draw above used ONLY sfh=4 (delayed-tau, SFR(t) ~ t*exp(-t/tau)) --
a single smooth, unimodal star formation history. Real galaxies, especially
low-mass/dwarf and actively star-forming systems, show genuinely bursty
SFHs: discrete episodes of enhanced star formation superimposed on a
smoother long-term trend. `bursty=True` below adds an opt-in, physically
motivated, literature-calibrated alternative; `bursty=False` (default)
leaves every existing call's output byte-for-byte unchanged.

Model: an Ornstein-Uhlenbeck (OU) process in the log-offset from the
smooth delayed-tau trend. Define

    Delta(t) = log10( SFR(t) / SFR_smooth(t) ),   SFR_smooth(t) = t*exp(-t/tau)

and let Delta(t) follow the stochastic differential equation

    d(Delta) = -(1/tau_OU) * Delta * dt + sigma_OU * sqrt(2/tau_OU) * dW(t)

where dW is a standard Wiener process. This is the Ornstein-Uhlenbeck
process (equivalently the Vasicek model in finance), and it is
simultaneously (a) a continuous-time, first-order MARKOV process -- its
future depends on the past only through its current value -- and (b) a
stationary GAUSSIAN PROCESS with an exponential (Matern-1/2) covariance
kernel, Cov(Delta(t), Delta(t')) = sigma_OU^2 * exp(-|t-t'|/tau_OU). These
are not two different candidate models to choose between; for this
problem they are the same object, which is why it is the standard tool in
the stochastic-SFH literature (see below) rather than an ad hoc choice.

The OU process has an EXACT (not finite-difference-approximate) discrete-
time transition for any time step dt, used directly here rather than an
Euler-Maruyama approximation (avoiding any discretization-error tuning):

    Delta_{i+1} | Delta_i ~ Normal( Delta_i * phi,  sigma_OU^2 * (1 - phi^2) ),
    phi = exp(-dt / tau_OU)

with stationary marginal Delta ~ Normal(0, sigma_OU^2) used as the initial
condition at the first grid point. SFR(t) = SFR_smooth(t) * 10**Delta(t) is
then automatically positive everywhere (no clipping needed, unlike an
additive discrete-burst construction), and setting sigma_OU=0 recovers
SFR(t) = SFR_smooth(t) exactly -- i.e. bursty SFH is a strict
generalization of the existing default, not a bolt-on.

The resulting tabulated (age, SFR) array is fed to FSPS via its OWN
literal non-parametric SFH mechanism: sfh=3 + StellarPopulation.
set_tabular_sfh(age, sfr) (piecewise-linear interpolation; verified
directly against python-fsps's source, github.com/dfm/python-fsps,
2026-08-31 -- no custom SFH integrator needed inside this module).

Why this design over a discrete-burst-count alternative: a simpler
"Poisson-distributed number of discrete top-hat bursts on a smooth base"
construction was also considered (and is itself real, citable practice --
some galaxy spectral-synthesis fitting codes parametrize up to ~6 discrete
bursts per SFH by age/metallicity/mass-fraction). It was not chosen
because (a) it requires ad hoc positivity clipping when a burst is added
on top of an already-positive base, (b) it is a patch on the existing
model rather than a strict generalization of it (no burst-rate limit
recovers the old default exactly the way sigma_OU->0 does here), and (c)
per PI direction (2026-08-31), a genuine statistical process was wanted
over a discrete-event bolt-on -- see this project's engineering log for
the full comparison presented before implementation.

Literature grounding for the two OU hyperparameters (see OU_SIGMA_RANGE_DEX/
OU_TAU_RANGE_GYR's own comments above for the itemized citation list):
Caplar & Tacchella (2019, MNRAS 487, 3845) built exactly this
stochastic-process model to explain the star-forming main sequence's
scatter, and fit it directly to real z~0 galaxies: sigma_OU is literally
the same quantity as the repeatedly-measured ~0.2-0.4 dex main-sequence
scatter (Brinchmann et al. 2004; Daddi et al. 2007; Noeske et al. 2007;
Whitaker et al. 2012; Speagle et al. 2014), and tau_OU is their fitted
correlation timescale (170 Myr, +169/-85, at z~0 M*~1e10 Msun),
consistent with the ~100 Myr (FIRE) to ~1000 Myr (IllustrisTNG) range
found in their companion simulation comparison and with the ~tens-of-Myr
burst durations / ~250 Myr recurrence independently inferred from
UV-vs-nebular-line studies of bursty low-mass galaxies (Weisz et al. 2012;
Guo et al. 2016; Emami et al. 2019; Broussard et al. 2019).

Why the OVERALL NORMALIZATION of the stochastic SFH doesn't need to be
physically calibrated: whatever SFH is used, the resulting composite
spectrum's absolute flux scale is already discarded downstream --
GALAXY.make_galaxy_templates() renormalizes every generated spectrum to
the requested apparent magnitude regardless of input scale (see this
module's own Returns docstring on baseflux, above). What a bursty SFH
changes -- and the actual physical point of adding it -- is the spectral
SHAPE (the young/old light mix), which feeds directly into diagnostics
like D4000 that this project's Dn4000-coupled-EW-scatter machinery
(Sec 1.4) already tracks; a bursty SFH supplies that axis with genuine
additional physical diversity rather than an arbitrary extra free
parameter.
"""
import os
import numpy as np
from astropy.table import Table


# ⚠ MAGIC: independent uniform priors over the stellar population
# parameters used to build the FSPS continuum library. No calibration data
# justifies these specific ranges; chosen to broadly span what real
# star-forming/quiescent galaxy populations occupy (e.g. Conroy 2013 ARAA
# review of stellar population synthesis parameter ranges in common use),
# modeled as independent free parameters per this project's established
# convention (same pattern as the Sec 1.2 auxiliary line-ratio priors) --
# real parameter *distributions* (and any covariance between them, e.g. the
# mass-metallicity relation) are intended to come later from the project's
# planned NPE/normalizing-flow fit to real spectra, not from hand-tuning
# these ranges further.
AGE_RANGE_GYR = (0.05, 13.0)      # ⚠ MAGIC
LOGZSOL_RANGE = (-1.0, 0.2)       # ⚠ MAGIC (solar units, log10(Z/Zsun))
DUST2_RANGE = (0.0, 1.0)          # ⚠ MAGIC (Calzetti-like V-band optical depth)
TAU_RANGE_GYR = (0.1, 10.0)       # ⚠ MAGIC (delayed-tau SFH e-folding time)

# ⚠ MAGIC: SFH functional form. sfh=4 is FSPS's "delayed tau" model
# (SFR(t) ~ t * exp(-t/tau)), a common simple default. This does not
# attempt to reproduce the real diversity of star formation histories
# (bursty, quenched-then-rejuvenated, etc.) that the real basis templates'
# underlying BC03 fits presumably capture -- flagged as a simplification
# for this first PR rather than guessing a more complex scheme unasked.
# RESOLVED (task #37, 2026-08-31): bursty=True below adds an opt-in,
# genuinely stochastic alternative -- see the "Bursty SFH" section further
# down this docstring for the full derivation and literature.
DEFAULT_SFH = 4
TABULAR_SFH = 3  # FSPS's non-parametric tabulated-SFH mode -- see set_tabular_sfh()
DEFAULT_DUST_TYPE = 2  # Calzetti (2000) attenuation curve

# ⚠ MAGIC (task #37, bursty SFH): Ornstein-Uhlenbeck process hyperparameter
# priors. NOT arbitrary -- both ranges are anchored to real measurements,
# same "independent uniform prior over a literature-supported range"
# convention as AGE_RANGE_GYR etc. above, real covariance still deferred to
# the project's planned NPE fit. See this module's "Bursty SFH" docstring
# section for the full derivation and citations.
#
# OU_SIGMA_RANGE_DEX: the star-forming-main-sequence scatter is repeatedly
# measured at sigma ~ 0.2-0.4 dex across independent studies and tracers
# (Brinchmann et al. 2004, MNRAS 351, 1151; Daddi et al. 2007, ApJ 670,
# 156; Noeske et al. 2007, ApJ 660, L43; Whitaker et al. 2012, ApJ 754,
# L29; Speagle et al. 2014, ApJS 214, 15) -- this IS the OU process's
# steady-state standard deviation under the Caplar & Tacchella (2019)
# framework (see below), not an independently-guessed number.
OU_SIGMA_RANGE_DEX = (0.2, 0.4)
# OU_TAU_RANGE_GYR: the process's correlation/"memory" timescale.
# Caplar & Tacchella (2019, MNRAS 487, 3845) fit this stochastic-process
# model directly to the observed z~0, M*~1e10 Msun main-sequence scatter
# and find tau_break = 170 (+169/-85) Myr; their companion simulation
# comparison (Iyer et al., in prep, cited therein) brackets tau_break
# between ~100 Myr (FIRE zoom-in simulations) and ~1000 Myr (IllustrisTNG),
# and UV-vs-nebular-line studies of bursty low-mass galaxies (Weisz et al.
# 2012, ApJ 744, 44; Guo et al. 2016, ApJ 833, 37; Emami et al. 2019, ApJ
# 881, 71; Broussard et al. 2019, ApJ 873, 74) independently find
# individual bursts of tens-of-Myr duration recurring on ~250 Myr periods
# -- consistent with the same 0.1-1.0 Gyr range.
OU_TAU_RANGE_GYR = (0.1, 1.0)
# Tabulation grid spacing fed to FSPS's sfh=3 (piecewise-linear
# interpolation). ~10x finer than OU_TAU_RANGE_GYR's lower bound (100 Myr)
# so the shortest allowed correlation timescale is comfortably resolved by
# the tabulated SFH FSPS actually integrates over.
OU_DT_GYR = 0.01

# Rest-frame wavelength window to keep from FSPS's native grid. Chosen to
# match the real DESI basis templates' own coverage (500-60000 Angstrom for
# ELG; see SETUP.md) for easier A/B comparison (Open Question 1's stated
# rationale for supplementing rather than replacing), and comfortably
# covers what's needed to redshift into DESI's 3600-9824 Angstrom observed
# window across this project's z ranges (down to rest ~1385 Angstrom for
# ELG's zmax=1.6).
DEFAULT_MINWAVE = 400.0
DEFAULT_MAXWAVE = 60000.0


def _continuum_flux_near(wave, flux, center, halfwidth=25.0):
    """Mean flux density in a narrow window around `center` [Angstrom].

    Used for OII_CONTINUUM/HBETA_CONTINUUM: unlike real-data continuum
    estimation (which must carefully avoid emission-line contamination),
    this is a pure stellar continuum with no emission lines yet added
    (EMSpectrum adds those later, downstream) -- so no line-avoidance
    sidebands are needed, just a small window to reduce sensitivity to any
    individual photospheric absorption feature landing exactly on center.
    """
    window = (wave >= center - halfwidth) & (wave <= center + halfwidth)
    if not np.any(window):
        # Fall back to nearest single pixel if the window is narrower than
        # the local grid spacing.
        idx = np.argmin(np.abs(wave - center))
        return float(flux[idx])
    return float(np.mean(flux[window]))


def _d4000(wave, flux):
    """Bruzual (1983) narrow D4000 break index.

    D4000 = <f_nu>[4000,4100] / <f_nu>[3850,3950], computed from f_lambda
    (FSPS's native peraa=True output) via f_nu = f_lambda * wave^2 / c.
    The proportionality constant in f_nu = f_lambda*wave**2/c cancels in
    the ratio, so c is omitted (arbitrary units are fine for a ratio).
    """
    fnu = flux * wave ** 2
    red = (wave >= 4000.0) & (wave <= 4100.0)
    blue = (wave >= 3850.0) & (wave <= 3950.0)
    if not np.any(red) or not np.any(blue):
        raise ValueError('Wavelength grid does not cover the D4000 sidebands '
                          '[3850,3950] and [4000,4100] Angstrom.')
    return float(np.mean(fnu[red]) / np.mean(fnu[blue]))


def _ou_log_offset(t_gyr, tau_ou_gyr, sigma_ou_dex, rand):
    """Exact discrete-time realization of a stationary Ornstein-Uhlenbeck
    process Delta(t), sampled at the (not necessarily uniformly spaced)
    times `t_gyr`. See this module's "Bursty SFH" docstring section for
    the full derivation and literature (Caplar & Tacchella 2019).

    Uses the EXACT OU transition kernel (not an Euler-Maruyama
    approximation), so accuracy does not depend on how finely `t_gyr` is
    sampled relative to tau_ou_gyr:

        Delta_{i+1} | Delta_i ~ Normal(Delta_i * phi, sigma_ou_dex^2 * (1-phi^2)),
        phi = exp(-(t_{i+1}-t_i) / tau_ou_gyr)

    with the stationary marginal Delta ~ Normal(0, sigma_ou_dex^2) used as
    the initial condition at t_gyr[0].

    Args:
        t_gyr (ndarray): strictly increasing time grid [Gyr], shape (n,).
        tau_ou_gyr (float): OU correlation timescale [Gyr].
        sigma_ou_dex (float): OU steady-state standard deviation [dex].
        rand (numpy.random.RandomState): source of randomness.

    Returns:
        ndarray, shape (n,): Delta(t) [dex], the log10 multiplicative
        offset from a smooth trend.
    """
    t_gyr = np.asarray(t_gyr, dtype=float)
    n = t_gyr.size
    delta = np.empty(n, dtype=float)
    delta[0] = rand.normal(0.0, sigma_ou_dex)
    for i in range(1, n):
        dt = t_gyr[i] - t_gyr[i - 1]
        phi = np.exp(-dt / tau_ou_gyr)
        delta[i] = delta[i - 1] * phi + rand.normal(0.0, sigma_ou_dex * np.sqrt(1.0 - phi ** 2))
    return delta


def _bursty_tabular_sfh(tage_gyr, tau_gyr, ou_tau_gyr, ou_sigma_dex, rand, dt_gyr=OU_DT_GYR):
    """Build a tabulated (age, SFR) array combining the existing smooth
    delayed-tau trend with an OU-process log-offset, suitable for FSPS's
    sfh=3 + set_tabular_sfh(). See this module's "Bursty SFH" docstring
    section for the full derivation and literature.

    Args:
        tage_gyr (float): population age [Gyr] -- the grid is built out to
            (and slightly past) this value.
        tau_gyr (float): delayed-tau e-folding time [Gyr] of the smooth
            trend SFR_smooth(t) = t*exp(-t/tau_gyr) (same physical
            parameter/meaning as the non-bursty sfh=4 path's `tau`).
        ou_tau_gyr, ou_sigma_dex (float): OU process hyperparameters, see
            _ou_log_offset().
        rand (numpy.random.RandomState): source of randomness.
        dt_gyr (float, optional): tabulation grid spacing [Gyr] (default
            OU_DT_GYR, ~10x finer than OU_TAU_RANGE_GYR's lower bound).

    Returns:
        Tuple of (age_gyr, sfr_msun_per_yr): both ndarrays, strictly
        increasing age, SFR strictly positive everywhere (satisfies
        set_tabular_sfh()'s own positivity assertions by construction).
    """
    n = max(int(np.ceil(tage_gyr / dt_gyr)) + 2, 3)
    age_gyr = dt_gyr * np.arange(1, n + 1)  # starts at dt_gyr, not 0 -- SFR_smooth(0)=0 exactly
    sfr_smooth = age_gyr * np.exp(-age_gyr / tau_gyr)
    delta = _ou_log_offset(age_gyr, ou_tau_gyr, ou_sigma_dex, rand)
    sfr = sfr_smooth * 10.0 ** delta
    # Numerical floor only -- satisfies set_tabular_sfh()'s "at least one
    # sfr > 1e-33" / "sfr cannot be negative" assertions with margin to
    # spare; physically negligible given sfr's actual scale.
    sfr = np.clip(sfr, 1e-20, None)
    return age_gyr, sfr


def _generate_shard(args):
    """Worker function for multiprocessing.Pool: generate one shard of the
    library in its own process (each process needs its own
    fsps.StellarPopulation -- FSPS's Fortran state is not safely shareable
    across processes).

    A module-level (not nested/closure) function is used deliberately so
    this is picklable regardless of the multiprocessing start method
    ('fork' on Linux would tolerate a closure too, but this keeps the code
    portable and easier to reason about).

    Returns (flux [n_in_shard, npix], basewave [npix]) -- basewave is
    returned from every shard (redundant across shards, but cheap and
    avoids needing a separate throwaway StellarPopulation in the parent
    process; it's deterministic given the compiled FSPS backend, so every
    shard necessarily agrees).
    """
    (tage, logzsol, dust2, tau, sfh, dust_type, minwave, maxwave,
     bursty, ou_tau, ou_sigma, ou_seeds, ou_dt_gyr) = args
    import fsps
    sp = fsps.StellarPopulation(zcontinuous=1, sfh=sfh, dust_type=dust_type,
                                 add_neb_emission=False, add_neb_continuum=False)
    wave_native = np.asarray(sp.wavelengths, dtype=float)
    mask = (wave_native >= minwave) & (wave_native <= maxwave)
    basewave = wave_native[mask]

    n = len(tage)
    flux = np.empty((n, mask.sum()), dtype=np.float64)
    for ii in range(n):
        sp.params['logzsol'] = logzsol[ii]
        sp.params['dust2'] = dust2[ii]
        if bursty:
            # tau[ii] still sets the underlying smooth trend's e-folding
            # time (see _bursty_tabular_sfh) -- its physical role as "how
            # quickly the long-term trend declines" is preserved, just
            # modulated by the OU-process log-offset on top.
            rand_ii = np.random.RandomState(ou_seeds[ii])
            age_grid, sfr_grid = _bursty_tabular_sfh(
                tage[ii], tau[ii], ou_tau[ii], ou_sigma[ii], rand_ii, dt_gyr=ou_dt_gyr)
            sp.params['sfh'] = TABULAR_SFH
            sp.set_tabular_sfh(age_grid, sfr_grid)
        else:
            sp.params['tau'] = tau[ii]
        _, spec = sp.get_spectrum(tage=tage[ii], peraa=True)
        flux[ii] = spec[mask]
    return flux, basewave


def fsps_basis_templates(objtype='ELG', nbase=128, minwave=DEFAULT_MINWAVE,
                          maxwave=DEFAULT_MAXWAVE, seed=None, sfh=DEFAULT_SFH,
                          dust_type=DEFAULT_DUST_TYPE,
                          age_range=AGE_RANGE_GYR, logzsol_range=LOGZSOL_RANGE,
                          dust2_range=DUST2_RANGE, tau_range=TAU_RANGE_GYR,
                          bursty=False, ou_tau_range=OU_TAU_RANGE_GYR,
                          ou_sigma_range=OU_SIGMA_RANGE_DEX, ou_dt_gyr=OU_DT_GYR,
                          nproc=1, verbose=False):
    """Build an FSPS-generated (baseflux, basewave, basemeta) triple with
    the same contract as desisim.io.read_basis_templates(), suitable for
    passing directly into GALAXY/ELG/BGS/LRG's baseflux/basewave/basemeta
    constructor arguments.

    Args:
        objtype (str, optional): 'ELG', 'BGS', or 'LRG' (default 'ELG').
            Determines which basemeta columns are populated with real
            (rather than placeholder) values: OII_CONTINUUM for ELG/LRG
            (normline='OII'/None), HBETA_CONTINUUM + HBETA_LIMIT for BGS
            (normline='HBETA'). D4000 is populated for all objtypes.
        nbase (int, optional): number of independent stellar population
            draws to generate (default 128). Each draw costs roughly
            0.7-0.8s in this environment (see module docstring Caveat 3);
            budget accordingly for large nbase.
        minwave, maxwave (float, optional): rest-frame wavelength window
            [Angstrom] to keep from FSPS's native grid (defaults
            400-60000 Angstrom, matching the real DESI basis templates'
            own coverage).
        seed (int, optional): random seed for the parameter draws.
        sfh, dust_type: passed to fsps.StellarPopulation (defaults: 4 =
            delayed-tau SFH, 2 = Calzetti attenuation). Ignored (overridden
            to TABULAR_SFH=3) if bursty=True -- see `bursty` below.
        age_range, logzsol_range, dust2_range, tau_range (tuple, optional):
            override the default (Gyr, dex, optical depth, Gyr) prior
            ranges (all ⚠ MAGIC; see module-level constants). tau_range
            still applies when bursty=True (it sets the underlying smooth
            trend's e-folding time -- see `bursty` below).
        bursty (bool, optional): default False (exact previous behavior --
            every draw uses the smooth sfh=4 delayed-tau model, byte-for-
            byte unchanged). If True, every draw instead uses an opt-in
            Ornstein-Uhlenbeck (OU) stochastic process superimposed (as a
            log10 multiplicative offset) on the same smooth delayed-tau
            trend, fed to FSPS via its own sfh=3 tabulated-SFH mechanism
            (StellarPopulation.set_tabular_sfh()). This is a strict
            generalization of the sfh=4 default: ou_sigma_range->(0,0)
            recovers it exactly. See this module's "Bursty SFH" docstring
            section for the full derivation and literature (Caplar &
            Tacchella 2019, MNRAS 487, 3845; Brinchmann et al. 2004; Daddi
            et al. 2007; Noeske et al. 2007; Whitaker et al. 2012; Speagle
            et al. 2014; Weisz et al. 2012; Guo et al. 2016; Emami et al.
            2019; Broussard et al. 2019).
        ou_tau_range (tuple, optional): OU process correlation timescale
            prior range [Gyr] (⚠ MAGIC, literature-anchored; default
            OU_TAU_RANGE_GYR = (0.1, 1.0)). Only used if bursty=True.
        ou_sigma_range (tuple, optional): OU process steady-state standard
            deviation prior range [dex] (⚠ MAGIC, literature-anchored;
            default OU_SIGMA_RANGE_DEX = (0.2, 0.4)). Only used if
            bursty=True.
        ou_dt_gyr (float, optional): tabulation grid spacing [Gyr] fed to
            FSPS's piecewise-linear sfh=3 interpolation (default
            OU_DT_GYR = 0.01, ~10x finer than ou_tau_range's lower bound).
            Only used if bursty=True.
        nproc (int, optional): number of worker processes (default 1 =
            single-threaded). Pass None to use os.cpu_count(). Each worker
            builds its own independent shard of `nbase` in its own
            fsps.StellarPopulation instance and results are concatenated;
            this is embarrassingly parallel since draws are independent.
            Measured cost in the sandbox this was developed in: ~0.7-2.3s
            per draw *per core* (see module docstring Caveat 3), so
            building a "tens of thousands"-scale library is a real
            multi-hour job on a 1-2 core machine but straightforwardly a
            few-minutes job on a real many-core node (e.g. a NERSC
            Perlmutter CPU node has 128 cores: ~20,000 draws / 128 * 1s
            ~ 156s). This was NOT run at that scale in this session -- the
            sandbox used to develop this has only 2 cores and no ability to
            run a job across tool-call boundaries, so this is a
            correctness-and-scaling-mechanism deliverable, not a completed
            tens-of-thousands library. Run it on an appropriately large
            machine for the real library build.
        verbose (bool, optional): print progress every 10 draws (nproc=1
            only; per-shard progress isn't aggregated across processes).

    Returns:
        Tuple of (baseflux, basewave, basemeta):
            baseflux (ndarray [nbase, npix]): rest-frame flux density
                [erg/s/cm2/A]. Absolute normalization is arbitrary --
                GALAXY.make_galaxy_templates() renormalizes every generated
                spectrum to the requested apparent magnitude regardless of
                input scale (see py/desisim/templates.py, `magnorm =
                10**(-0.4*mag) / normmaggies`), so this doesn't need to
                match the real basis templates' absolute flux scale.
            basewave (ndarray [npix]): rest-frame wavelength [Angstrom].
            basemeta (astropy.Table [nbase]): TEMPLATEID, D4000, and
                (objtype-dependent) OII_CONTINUUM / HBETA_CONTINUUM /
                HBETA_LIMIT, plus the generating parameters (AGE_GYR,
                LOGZSOL, DUST2, TAU_GYR, BURSTY, OU_TAU_GYR, OU_SIGMA_DEX)
                for transparency/debugging (not read by GALAXY, but useful
                ground truth to keep around given this whole project is
                about having known ground truth for training data).
                OU_TAU_GYR/OU_SIGMA_DEX are NaN when bursty=False (not
                applicable).
    """
    objtype = objtype.upper()
    if objtype not in ('ELG', 'BGS', 'LRG'):
        raise ValueError("objtype must be one of 'ELG', 'BGS', 'LRG'; got {!r}".format(objtype))

    rand = np.random.RandomState(seed)
    tage = rand.uniform(age_range[0], age_range[1], nbase)
    logzsol = rand.uniform(logzsol_range[0], logzsol_range[1], nbase)
    dust2 = rand.uniform(dust2_range[0], dust2_range[1], nbase)
    tau = rand.uniform(tau_range[0], tau_range[1], nbase)

    # Bursty-mode draws come AFTER the four draws above, so a bursty=False
    # call's tage/logzsol/dust2/tau draws are byte-for-byte identical to
    # what they'd be with bursty=True for the same seed (nothing about
    # enabling bursty perturbs the RNG stream position for draws that
    # already existed before this feature).
    if bursty:
        ou_tau = rand.uniform(ou_tau_range[0], ou_tau_range[1], nbase)
        ou_sigma = rand.uniform(ou_sigma_range[0], ou_sigma_range[1], nbase)
        ou_seeds = rand.randint(0, 2**31 - 1, nbase)
        effective_sfh = TABULAR_SFH
    else:
        ou_tau = np.full(nbase, np.nan)
        ou_sigma = np.full(nbase, np.nan)
        ou_seeds = np.zeros(nbase, dtype=np.int64)
        effective_sfh = sfh

    if nproc is None:
        nproc = os.cpu_count() or 1
    nproc = max(1, min(int(nproc), nbase))

    if nproc == 1:
        baseflux, basewave = _generate_shard((tage, logzsol, dust2, tau, effective_sfh, dust_type, minwave, maxwave,
                                               bursty, ou_tau, ou_sigma, ou_seeds, ou_dt_gyr))
        if verbose:
            print('fsps_basis_templates: generated {} templates (nproc=1)'.format(nbase))
    else:
        import multiprocessing
        shard_idx = np.array_split(np.arange(nbase), nproc)
        shard_args = [(tage[idx], logzsol[idx], dust2[idx], tau[idx], effective_sfh, dust_type,
                       minwave, maxwave, bursty, ou_tau[idx], ou_sigma[idx], ou_seeds[idx], ou_dt_gyr)
                      for idx in shard_idx]
        if verbose:
            print('fsps_basis_templates: dispatching {} draws across {} worker processes'.format(nbase, nproc))
        with multiprocessing.Pool(nproc) as pool:
            shard_results = pool.map(_generate_shard, shard_args)
        baseflux = np.concatenate([flux for flux, _ in shard_results], axis=0)
        basewave = shard_results[0][1]
        # All shards must agree on the wavelength grid (it's deterministic
        # given the compiled FSPS backend, independent of which draws a
        # given shard happened to receive) -- verify rather than assume,
        # since a silent mismatch here would misalign flux rows.
        for _, shard_wave in shard_results[1:]:
            if not np.array_equal(shard_wave, basewave):
                raise RuntimeError('fsps_basis_templates: worker processes disagree on the native FSPS '
                                    'wavelength grid; this should be impossible for a fixed compiled '
                                    'backend and indicates environment inconsistency across processes.')

    if not np.all(np.isfinite(baseflux)) or np.any(baseflux < 0):
        raise RuntimeError('fsps_basis_templates: FSPS returned non-finite or negative flux; '
                            'this should not happen for a physical SED and indicates a real bug, '
                            'not an expected edge case.')

    d4000 = np.array([_d4000(basewave, baseflux[ii]) for ii in range(nbase)])
    oii_continuum = np.array([_continuum_flux_near(basewave, baseflux[ii], 3727.4) for ii in range(nbase)])
    hbeta_continuum = np.array([_continuum_flux_near(basewave, baseflux[ii], 4861.3) for ii in range(nbase)])

    basemeta = Table()
    basemeta['TEMPLATEID'] = np.arange(nbase, dtype=np.int32)
    basemeta['D4000'] = d4000.astype('f4')
    basemeta['OII_CONTINUUM'] = oii_continuum.astype('f4')
    basemeta['HBETA_CONTINUUM'] = hbeta_continuum.astype('f4')
    # All-zero HBETA_LIMIT: every synthetic template is a "valid measurement"
    # in the sense make_galaxy_templates checks (self.basemeta['HBETA_LIMIT']
    # == 0); there's no real-data upper-limit concept for a pure forward
    # model. Flagging this as a simplification, not a physical claim.
    basemeta['HBETA_LIMIT'] = np.zeros(nbase, dtype=np.int32)
    basemeta['AGE_GYR'] = tage.astype('f4')
    basemeta['LOGZSOL'] = logzsol.astype('f4')
    basemeta['DUST2'] = dust2.astype('f4')
    basemeta['TAU_GYR'] = tau.astype('f4')
    basemeta['BURSTY'] = np.full(nbase, bursty, dtype=bool)
    basemeta['OU_TAU_GYR'] = ou_tau.astype('f4')
    basemeta['OU_SIGMA_DEX'] = ou_sigma.astype('f4')

    return baseflux, basewave, basemeta
