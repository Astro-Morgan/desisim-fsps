import numpy as np

from demiurge.quasar_continuum.comptonization import donthcomp


def test_donthcomp_produces_finite_nonnegative_shape():
    ear = np.geomspace(1e-4, 200.0, 301)
    shape = donthcomp(ear, gamma=2.5, kte_kev=0.2, kt_seed_kev=10.0)
    assert shape.shape == (300,)
    assert np.all(np.isfinite(shape))
    assert np.all(shape >= 0.0)
    assert shape.sum() > 0.0


def test_donthcomp_harder_electron_temperature_extends_tail_further():
    ear = np.geomspace(1e-4, 200.0, 301)
    en_mid = np.sqrt(ear[:-1] * ear[1:])

    cool = donthcomp(ear, gamma=2.0, kte_kev=20.0, kt_seed_kev=1.0)
    hot = donthcomp(ear, gamma=2.0, kte_kev=100.0, kt_seed_kev=1.0)

    high_e_mask = en_mid > 30.0
    assert hot[high_e_mask].sum() > cool[high_e_mask].sum()
