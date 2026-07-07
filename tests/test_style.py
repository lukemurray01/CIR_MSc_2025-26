from src.utils.style import (
    DEFAULT_LINEWIDTH,
    METHOD_COLOURS,
    METHOD_LABELS,
    METHOD_LINESTYLES,
    REGIME_COLOURS,
)


def test_regime_colours():
    assert set(REGIME_COLOURS.keys()) == {"A", "B", "C", "D", "E"}


def test_method_styles_exist_for_registered_methods():
    expected = {"FTE", "HH", "ProjEuler", "KL", "KLM", "BLT", "Exact"}
    assert expected.issubset(METHOD_COLOURS.keys())
    assert expected.issubset(METHOD_LINESTYLES.keys())
    assert expected.issubset(METHOD_LABELS.keys())


def test_default_linewidth_is_spelled_correctly():
    assert DEFAULT_LINEWIDTH > 0.0
