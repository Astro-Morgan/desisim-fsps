#!/usr/bin/env python
"""
visual_check_mock_generation.py
================================

A visual, qualitative sanity check of this fork's new mock-generation
components (Sec 1.2/1.3 tunable + new emission lines, non-stellar ISM/CGM
absorption, and the Legendre-polynomial camera-calibration artifact),
run before continuing further pipeline work.

This is deliberately NOT a survey-realistic pipeline run: it builds one
rest-frame continuum from a real DESI basis template, adds each new
component on top of it "by hand" (rather than through GALAXY.make_templates(),
which does not yet wire these newer components together -- see each
module's docstring), and plots the sum. Flux normalization is illustrative
only (no distance-modulus/magnitude calibration); the point is to see how
each generating parameter changes the *shape* of the spectrum, not to
produce a survey-accurate flux scale.

Runs identically here (sandbox, using the manually-uploaded local copies of
the real v3.1 DESI basis templates) and on Perlmutter (set
$DESI_BASIS_TEMPLATES to the real NERSC path, e.g.
/global/cfs/cdirs/desi/spectro/templates/basis_templates/v3.1, and this
script needs no changes).

Usage:
    export DESI_BASIS_TEMPLATES=/path/to/basis_templates
    python bin/visual_check_mock_generation.py [--outdir DIR] [--objtype ELG]
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from desisim.io import read_basis_templates
from desisim.templates import EMSpectrum
from desisim.absorption import AbsorptionSpectrum
from desisim.camera_calibration import CameraCalibration
from desisim.dust import DustAttenuation


def build_rest_frame_mock(basewave, base_continuum, em_kwargs, ab_kwargs, seed):
    """Sum one real continuum + EMSpectrum + AbsorptionSpectrum on the
    continuum's own rest-frame wavelength grid. Returns (wave, continuum,
    emission, absorption, total) -- all four kept separate so the plots
    below can show ground-truth decomposition, not just the sum.
    """
    em = EMSpectrum(log10wave=np.log10(basewave),
                     include_mgii=em_kwargs.pop('include_mgii', False),
                     include_new_lines=em_kwargs.pop('include_new_lines', False))
    emspec, ewave, _ = em.spectrum(seed=seed, **em_kwargs)

    ab = AbsorptionSpectrum(log10wave=np.log10(basewave),
                             include_lines=ab_kwargs.pop('include_lines', None))
    absflux, awave, _ = ab.spectrum(basewave, base_continuum, seed=seed, **ab_kwargs)

    total = base_continuum + emspec + absflux
    return basewave, base_continuum, emspec, absflux, total


def to_observed_frame(basewave, flux, z):
    """Simple rest -> observed frame stretch (wave *= 1+z). Flux is left
    on the same per-unit-rest-wavelength scale (no distance-modulus/
    (1+z)-dimming applied) -- this is a qualitative shape check, not a
    survey-flux-accurate transform."""
    return basewave * (1.0 + z), flux


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='.', help='Directory to write PNG figures to')
    parser.add_argument('--objtype', default='ELG', choices=['ELG', 'BGS', 'LRG'])
    parser.add_argument('--templateid', type=int, default=0,
                         help='Index of the real basis-template continuum to use')
    parser.add_argument('--redshift', type=float, default=0.8)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print('Reading real {} basis templates from $DESI_BASIS_TEMPLATES={}'.format(
        args.objtype, os.environ.get('DESI_BASIS_TEMPLATES', '<not set>')))
    baseflux, basewave, basemeta = read_basis_templates(objtype=args.objtype)
    base_continuum = baseflux[args.templateid]
    print('Using real template index {}: D4000={:.3f}, OII_CONTINUUM={:.3g}, HBETA_CONTINUUM={:.3g}'.format(
        args.templateid, basemeta['D4000'][args.templateid],
        basemeta['OII_CONTINUUM'][args.templateid],
        basemeta.get('HBETA_CONTINUUM', basemeta['OII_CONTINUUM'])[args.templateid]
        if 'HBETA_CONTINUUM' in basemeta.colnames else float('nan')))

    # Illustrative-only EW*continuum-density construction (not the real
    # D4000-coupled draw GALAXY.make_galaxy_templates uses internally --
    # this script builds spectra by hand from the standalone component
    # classes, which don't know about D4000 at all). EW=80A is a
    # deliberately strong/visible value for this plot, not a typical one.
    hbetaflux = 80.0 * basemeta['OII_CONTINUUM'][args.templateid]

    fixed_ratios = dict(oiiihbeta=-0.2, oiihbeta=0.1, niihbeta=-0.2, siihbeta=-0.3)

    # ------------------------------------------------------------------
    # Figure 1: legacy emission vs. Sec 1.3 new lines (narrow+broad), zoomed
    # on the new-line-rich 3800-5000A rest-frame region.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    wave0, cont0, em0, ab0, tot0 = build_rest_frame_mock(
        basewave, base_continuum,
        dict(hbetaflux=hbetaflux, **fixed_ratios), dict(tau0={n: 0.0 for n in AbsorptionSpectrum.LINE_NAMES}),
        seed=1)

    new_line_ratios = {n: 0.3 for n in EMSpectrum.NEW_LINE_NAMES}       # exaggerated for visibility
    new_line_broad_ratios = {n: 0.15 for n in EMSpectrum.NEW_LINE_NAMES}
    wave1, cont1, em1, ab1, tot1 = build_rest_frame_mock(
        basewave, base_continuum,
        dict(hbetaflux=hbetaflux, include_new_lines=True, new_line_ratios=new_line_ratios,
             new_line_broad_ratios=new_line_broad_ratios, broadsigma=1500.0, **fixed_ratios),
        dict(tau0={n: 0.0 for n in AbsorptionSpectrum.LINE_NAMES}),
        seed=1)

    zoom = (basewave > 3800) & (basewave < 5050)
    axes[0].plot(wave0[zoom], tot0[zoom], color='k', lw=1, label='Legacy (include_new_lines=False)')
    axes[0].plot(wave1[zoom], tot1[zoom], color='crimson', lw=1, label='Sec 1.3 new lines ON (exaggerated ratios)')
    axes[0].set_ylabel('Flux + continuum\n[arb., rest frame]')
    axes[0].set_title('{}: legacy vs. new narrow+broad lines (template #{}, D4000={:.2f})'.format(
        args.objtype, args.templateid, basemeta['D4000'][args.templateid]))
    axes[0].legend(loc='upper right', fontsize=9)

    axes[1].plot(wave1[zoom], em1[zoom], color='crimson', lw=1, label='Emission only (new lines ON)')
    reference_em = EMSpectrum(include_new_lines=True)
    for name in EMSpectrum.NEW_LINE_NAMES:
        row = reference_em.line[reference_em.line['name'] == name]
        if len(row) and 3800 < row['wave'][0] < 5050:
            axes[1].axvline(row['wave'][0], color='gray', ls=':', lw=0.7)
            axes[1].text(row['wave'][0], axes[1].get_ylim()[1]*0.9, name, rotation=90,
                         fontsize=7, ha='right', va='top')
    axes[1].set_xlabel('Rest-frame wavelength [Angstrom]')
    axes[1].set_ylabel('Emission-line flux\n[erg/s/cm2/A, rest]')
    axes[1].legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    fig1_path = os.path.join(args.outdir, '1_new_emission_lines.png')
    fig.savefig(fig1_path, dpi=130)
    plt.close(fig)
    print('Wrote', fig1_path)

    # ------------------------------------------------------------------
    # Figure 2: broadsigma sweep on one line (HeII 4686), narrow ratio
    # fixed, to show the narrow/broad width contrast directly.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    zoom2 = (basewave > 4550) & (basewave < 4850)
    for broadsigma, color in [(500.0, 'tab:blue'), (1500.0, 'tab:green'), (4000.0, 'tab:red')]:
        w, c, e, a, t = build_rest_frame_mock(
            basewave, base_continuum,
            dict(hbetaflux=hbetaflux, include_new_lines=True,
                 new_line_ratios={'HeII_4686': 0.15}, new_line_broad_ratios={'HeII_4686': 0.4},
                 broadsigma=broadsigma, **fixed_ratios),
            dict(tau0={n: 0.0 for n in AbsorptionSpectrum.LINE_NAMES}),
            seed=1)
        ax.plot(w[zoom2], e[zoom2], color=color, lw=1.2, label='broadsigma={:.0f} km/s'.format(broadsigma))
    ax.set_xlabel('Rest-frame wavelength [Angstrom]')
    ax.set_ylabel('Emission-line flux [erg/s/cm2/A, rest]')
    ax.set_title('HeII 4686 narrow+broad profile vs. broadsigma (fixed ratios)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig2_path = os.path.join(args.outdir, '2_broadsigma_sweep.png')
    fig.savefig(fig2_path, dpi=130)
    plt.close(fig)
    print('Wrote', fig2_path)

    # ------------------------------------------------------------------
    # Figure 3: ISM/CGM absorption off vs. on, zoomed on each of the 3
    # line complexes (CaII H&K, NaID, MgII).
    # ------------------------------------------------------------------
    strong_tau0 = {n: 1.5 for n in AbsorptionSpectrum.LINE_NAMES}
    w_off, c_off, e_off, a_off, t_off = build_rest_frame_mock(
        basewave, base_continuum, dict(hbetaflux=hbetaflux, **fixed_ratios),
        dict(tau0={n: 0.0 for n in AbsorptionSpectrum.LINE_NAMES}), seed=1)
    w_on, c_on, e_on, a_on, t_on = build_rest_frame_mock(
        basewave, base_continuum, dict(hbetaflux=hbetaflux, **fixed_ratios),
        dict(tau0=strong_tau0, sigma_kms=80.0), seed=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    windows = [('Ca II H&K', 3880, 4020), ('Na I D', 5830, 5960), ('Mg II 2796/2803', 2740, 2860)]
    for ax, (title, lo, hi) in zip(axes, windows):
        m = (basewave > lo) & (basewave < hi)
        ax.plot(w_off[m], t_off[m], color='k', lw=1, label='Absorption OFF')
        ax.plot(w_on[m], t_on[m], color='tab:purple', lw=1, label='Absorption ON (tau0=1.5)')
        ax.set_title(title)
        ax.set_xlabel('Rest wave [A]')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Flux [arb., rest frame]')
    fig.suptitle('{}: ISM/CGM absorption channel, template #{}'.format(args.objtype, args.templateid))
    fig.tight_layout()
    fig3_path = os.path.join(args.outdir, '3_ism_absorption.png')
    fig.savefig(fig3_path, dpi=130)
    plt.close(fig)
    print('Wrote', fig3_path)

    # ------------------------------------------------------------------
    # Figure 4: full observed-frame mock, camera-calibration artifact off
    # vs. on (exaggerated coefficients for visibility), with camera
    # boundaries marked.
    # ------------------------------------------------------------------
    z = args.redshift
    wave_obs, flux_obs = to_observed_frame(w_on, t_on, z)
    desi_window = (wave_obs > 3600.0) & (wave_obs < 9824.0)
    wave_obs, flux_obs = wave_obs[desi_window], flux_obs[desi_window]

    cc = CameraCalibration()
    exaggerated_coeffs = {
        'b': np.array([0.0, 0.0]),
        'r': np.array([2.0, -3.0]) * np.median(np.abs(flux_obs)),
        'z': np.array([-1.5, 1.0]) * np.median(np.abs(flux_obs)),
    }
    calib_flux, _, coeff_table = cc.spectrum(wave_obs, coeffs=exaggerated_coeffs)
    flux_obs_calibrated = flux_obs + calib_flux

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(wave_obs, flux_obs, color='k', lw=0.8, label='No calibration artifact')
    ax.plot(wave_obs, flux_obs_calibrated, color='tab:orange', lw=0.8, alpha=0.85,
            label='+ Legendre calibration artifact (exaggerated)')
    bounds = cc._camera_assignment_boundaries()
    for cam, (lo, hi) in bounds.items():
        ax.axvline(lo, color='gray', ls='--', lw=0.6)
    ax.set_xlabel('Observed wavelength [Angstrom]')
    ax.set_ylabel('Flux [arb.]')
    ax.set_title('{} at z={:.2f}: full mock (continuum+emission+absorption) with/without camera calibration artifact\n'
                 '(dashed lines = b/r/z camera assignment boundaries)'.format(args.objtype, z))
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig4_path = os.path.join(args.outdir, '4_full_mock_with_calibration.png')
    fig.savefig(fig4_path, dpi=130)
    plt.close(fig)
    print('Wrote', fig4_path)
    print(coeff_table)

    # ------------------------------------------------------------------
    # Figure 5: additive component decomposition -- continuum, emission,
    # absorption, and their sum, stacked so each channel's own ground-truth
    # contribution is directly visible (this is the actual point of the
    # whole project: mocks that decompose cleanly into known components).
    # No dust panel yet -- see bin/README or the PR discussion for the
    # dust-attenuation open question; a 5th panel is reserved for it.
    # ------------------------------------------------------------------
    strong_new_ratios = {n: 0.3 for n in EMSpectrum.NEW_LINE_NAMES}
    strong_new_broad = {n: 0.15 for n in EMSpectrum.NEW_LINE_NAMES}
    strong_tau0_decomp = {n: 1.0 for n in AbsorptionSpectrum.LINE_NAMES}

    w_d, cont_d, em_d, ab_d, tot_d = build_rest_frame_mock(
        basewave, base_continuum,
        dict(hbetaflux=hbetaflux, include_new_lines=True, new_line_ratios=strong_new_ratios,
             new_line_broad_ratios=strong_new_broad, broadsigma=1500.0, **fixed_ratios),
        dict(tau0=strong_tau0_decomp, sigma_kms=80.0),
        seed=1)

    # Dust attenuation applies to (continuum+emission) as a single
    # intrinsic SED -- see dust.py's module docstring for why (both are
    # assumed to sit behind the same foreground dust column, unlike
    # AbsorptionSpectrum which by design only attenuates the continuum).
    dust = DustAttenuation()
    intrinsic_d = cont_d + em_d
    strong_theta = dict(theta0=1.2, theta1=1.0, theta2=0.6, theta3=0.1)  # exaggerated for visibility
    dust_flux_d, _, dust_table = dust.spectrum(w_d, intrinsic_d, theta=strong_theta, seed=1)
    tot_d_with_dust = tot_d + dust_flux_d

    wave_obs_d, cont_obs_d = to_observed_frame(w_d, cont_d, z)
    _, em_obs_d = to_observed_frame(w_d, em_d, z)
    _, ab_obs_d = to_observed_frame(w_d, ab_d, z)
    _, dust_obs_d = to_observed_frame(w_d, dust_flux_d, z)
    _, tot_obs_d = to_observed_frame(w_d, tot_d_with_dust, z)
    window_d = (wave_obs_d > 3600.0) & (wave_obs_d < 9824.0)

    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
    panels = [
        ('Continuum\n(real basis template)', cont_obs_d, 'tab:blue'),
        ('+ Emission\n(EMSpectrum, new lines ON)', em_obs_d, 'tab:red'),
        ('+ Absorption\n(AbsorptionSpectrum, ISM/CGM)', ab_obs_d, 'tab:purple'),
        ('+ Dust\n(DustAttenuation, exaggerated theta)', dust_obs_d, 'tab:brown'),
        ('= Total mock\n(all 4 channels summed)', tot_obs_d, 'k'),
    ]
    for ax, (label, arr, color) in zip(axes, panels):
        ax.plot(wave_obs_d[window_d], arr[window_d], color=color, lw=0.8)
        ax.set_ylabel(label, fontsize=9)
        ax.axhline(0.0, color='gray', lw=0.5, ls=':')
    axes[-1].set_xlabel('Observed wavelength [Angstrom]')
    fig.suptitle('{} at z={:.2f}: additive component decomposition (template #{})'.format(
        args.objtype, z, args.templateid))
    print(dust_table)
    fig.tight_layout()
    fig5_path = os.path.join(args.outdir, '5_additive_decomposition.png')
    fig.savefig(fig5_path, dpi=130)
    plt.close(fig)
    print('Wrote', fig5_path)

    print('\nDone. All figures written to', os.path.abspath(args.outdir))


if __name__ == '__main__':
    main()
