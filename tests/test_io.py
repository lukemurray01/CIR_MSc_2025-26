# tests/test_io.py
from src.utils.io import (
    CONFIGS_DIR, FIGURES_DIR, RESULTS_DIR,
    config_path, figure_path, results_path,
    ensure_dirs,
)

def test_figure_path_returns_correct_path():
    path = figure_path("test_figure.pdf")
    assert path.parent == FIGURES_DIR
    assert path.name == "test_figure.pdf"

def test_results_path_returns_correct_path():
    path = results_path("test_results.csv")
    assert path.parent == RESULTS_DIR
    assert path.name == "test_results.csv"

def test_config_path_points_to_configs_directory():
    path = config_path("regimes.yaml")
    assert path.parent == CONFIGS_DIR
    assert path.name == "regimes.yaml"

def test_ensure_dirs_creates_standard_directories():
    ensure_dirs()
    assert FIGURES_DIR.exists()
    assert RESULTS_DIR.exists()