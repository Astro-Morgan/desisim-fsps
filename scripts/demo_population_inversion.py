"""
Demonstration/visual-verification script for
`demiurge.galaxy_continuum.population_inversion` -- NOT part of the
installed package, matching this project's own `scripts/` convention
(auxiliary one-off tools, see BUILD.md).

Builds a real dense-grid + Monte-Carlo target grid, queries it against
three illustrative literature-motivated targets (a thin-disc-like, a
halo-like, and a massive-elliptical-like population), and produces a plot
showing the queried neighborhoods actually land where expected -- this
project's own standing discipline is to check things visually, not just
trust that a mechanism "looks right" from its own unit tests (see e.g.
the reddening-floor saga, [[project-blending-reddening-channel]]).

Unit-conversion caveat, stated plainly rather than silently assumed: the
literature targets pulled during planning (Bensby et al. 2014 for disc,
An et al. 2013 / Xue et al. 2015 for halo, Thomas et al. 2005 for
elliptical) are [Fe/H]/[Z/H] values relative to solar -- this project's
own `Z(t) = yield * ln(1/mu(t))` closed-box relation produces an ABSOLUTE
metal-mass fraction, with no solar-metallicity convention adopted
anywhere yet. This script picks Z_sun = 0.0142 (Asplund, Grevesse, Sauval
& Scott 2009) purely to make an illustrative, approximate comparison
possible -- this is NOT a project-wide decision about which solar
reference to adopt (that translation problem is still open, per project
memory), just a labeled assumption local to this demo.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from demiurge.galaxy_continuum.population_inversion import build_population_target_grid

Z_SUN_ASPLUND2009 = 0.0142

TARGETS = {
    "thin disc (Bensby+14-ish)": dict(feh=-0.1, age_gyr=4.0, color="tab:blue"),
    "halo (Xue+15)": dict(feh=-1.75, age_gyr=12.0, color="tab:orange"),
    "massive elliptical (Thomas+05)": dict(feh=0.30, age_gyr=10.0, color="tab:red"),
}


def main() -> None:
    rng = np.random.default_rng(0)
    target_z_values = [Z_SUN_ASPLUND2009 * 10.0 ** spec["feh"] for spec in TARGETS.values()]
    print("Building the population target grid (6D, n_grid_points_per_axis=6, n_monte_carlo=5000, "
          "supplemented with targeted samples around each of the 3 targets below)...")
    grid = build_population_target_grid(
        rng=rng, n_grid_points_per_axis=6, n_monte_carlo=5000,
        target_z_values=target_z_values, n_yield_efficiency_points_per_target=50, n_shape_draws_per_target_point=20,
    )
    print(f"Grid built: {len(grid)} points "
          f"(age range {grid.mass_weighted_age_gyr.min():.2f}-{grid.mass_weighted_age_gyr.max():.2f} Gyr, "
          f"Z range {grid.mass_weighted_z.min():.2e}-{grid.mass_weighted_z.max():.2e})")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(grid.mass_weighted_age_gyr, grid.mass_weighted_z, s=4, alpha=0.15, color="0.5", label="full grid")

    for label, spec in TARGETS.items():
        target_z = Z_SUN_ASPLUND2009 * 10.0 ** spec["feh"]
        target_age = spec["age_gyr"]
        idx = grid.nearest(target_age_gyr=target_age, target_z=target_z, k=100)

        yields = grid.yield_[idx]
        efficiencies = grid.star_formation_efficiency[idx]
        ages = grid.mass_weighted_age_gyr[idx]
        zs = grid.mass_weighted_z[idx]

        print(f"\n[{label}] target: age={target_age:.1f} Gyr, [Fe/H]-equiv={spec['feh']:+.2f} -> Z={target_z:.4e}")
        print(f"  nearest {len(idx)} neighbors achieve age={ages.mean():.2f}+/-{ages.std():.2f} Gyr, "
              f"Z={np.median(zs):.4e} (range {zs.min():.4e}-{zs.max():.4e})")
        print(f"  implied yield: median={np.median(yields):.4e}, range {yields.min():.4e}-{yields.max():.4e}")
        print(f"  implied star_formation_efficiency: median={np.median(efficiencies):.3f}, "
              f"range {efficiencies.min():.3f}-{efficiencies.max():.3f}")

        ax.scatter(ages, zs, s=14, alpha=0.7, color=spec["color"], label=f"{label} (k=100 nearest)")
        ax.scatter([target_age], [target_z], marker="*", s=300, color=spec["color"],
                   edgecolor="black", linewidth=1.2, zorder=5)

    ax.set_yscale("log")
    ax.set_xlabel("mass-weighted age [Gyr]")
    ax.set_ylabel("mass-weighted Z (absolute metal mass fraction)")
    ax.set_title("population_inversion grid: full cloud + 3 illustrative target queries")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path = "scripts/demo_population_inversion.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
