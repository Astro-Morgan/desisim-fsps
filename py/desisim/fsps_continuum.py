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
DEFAULT_SFH = 4
DEFAULT_DUST_TYPE = 2  # Calzetti (2000) attenuation curve

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
    tage, logzsol, dust2, tau, sfh, dust_type, minwave, maxwave = args
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
        sp.params['tau'] = tau[ii]
        _, spec = sp.get_spectrum(tage=tage[ii], peraa=True)
        flux[ii] = spec[mask]
    return flux, basewave


def fsps_basis_templates(objtype='ELG', nbase=128, minwave=DEFAULT_MINWAVE,
                          maxwave=DEFAULT_MAXWAVE, seed=None, sfh=DEFAULT_SFH,
                          dust_type=DEFAULT_DUST_TYPE,
                          age_range=AGE_RANGE_GYR, logzsol_range=LOGZSOL_RANGE,
                          dust2_range=DUST2_RANGE, tau_range=TAU_RANGE_GYR,
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
            delayed-tau SFH, 2 = Calzetti attenuation).
        age_range, logzsol_range, dust2_range, tau_range (tuple, optional):
            override the default (Gyr, dex, optical depth, Gyr) prior
            ranges (all ⚠ MAGIC; see module-level constants).
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
                LOGZSOL, DUST2, TAU_GYR) for transparency/debugging (not
                read by GALAXY, but useful ground truth to keep around
                given this whole project is about having known ground
                truth for training data).
    """
    objtype = objtype.upper()
    if objtype not in ('ELG', 'BGS', 'LRG'):
        raise ValueError("objtype must be one of 'ELG', 'BGS', 'LRG'; got {!r}".format(objtype))

    rand = np.random.RandomState(seed)
    tage = rand.uniform(age_range[0], age_range[1], nbase)
    logzsol = rand.uniform(logzsol_range[0], logzsol_range[1], nbase)
    dust2 = rand.uniform(dust2_range[0], dust2_range[1], nbase)
    tau = rand.uniform(tau_range[0], tau_range[1], nbase)

    if nproc is None:
        nproc = os.cpu_count() or 1
    nproc = max(1, min(int(nproc), nbase))

    if nproc == 1:
        baseflux, basewave = _generate_shard((tage, logzsol, dust2, tau, sfh, dust_type, minwave, maxwave))
        if verbose:
            print('fsps_basis_templates: generated {} templates (nproc=1)'.format(nbase))
    else:
        import multiprocessing
        shard_idx = np.array_split(np.arange(nbase), nproc)
        shard_args = [(tage[idx], logzsol[idx], dust2[idx], tau[idx], sfh, dust_type, minwave, maxwave)
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

    return baseflux, basewave, basemeta
