from cog_ew.gan_signals.sampler import build_type_catalog


def test_catalog_has_known_first_then_interpolated():
    catalog = build_type_catalog(8, alphas=(0.25, 0.5, 0.75), extrapolate=False)
    assert len(catalog) >= 50
    known = catalog[:8]
    assert all(t.is_known and t.alpha == 0.0 and t.source_a == t.source_b for t in known)
    assert [t.source_a for t in known] == list(range(8))
    novel = catalog[8:]
    assert all(not t.is_known and t.source_a < t.source_b for t in novel)
    assert all(t.alpha in (0.25, 0.5, 0.75) for t in novel)
    assert [t.type_id for t in catalog] == list(range(len(catalog)))


def test_catalog_is_deterministic():
    a = build_type_catalog(8, alphas=(0.25, 0.5, 0.75))
    b = build_type_catalog(8, alphas=(0.25, 0.5, 0.75))
    assert a == b


def test_catalog_extrapolate_adds_out_of_range_alphas():
    catalog = build_type_catalog(8, alphas=(0.5,), extrapolate=True)
    novel_alphas = {t.alpha for t in catalog if not t.is_known}
    assert -0.25 in novel_alphas and 1.25 in novel_alphas
