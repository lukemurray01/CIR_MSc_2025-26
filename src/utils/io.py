# Input output helpers for repo
# Functions are for automating file saves for figures and results

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTES_DIR = PROJECT_ROOT / "notes"
CONFIGS_DIR = PROJECT_ROOT / "configs"


# Create the directory if it doesn't already exist
def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Create the standard project output directories
def ensure_dirs() -> None:
    ensure_dir(FIGURES_DIR)
    ensure_dir(RESULTS_DIR)
    ensure_dir(NOTES_DIR)
    ensure_dir(CONFIGS_DIR)


# Return a path inside the figures directory
def figure_path(filename: str) -> Path:
    ensure_dir(FIGURES_DIR)
    return FIGURES_DIR / filename


# Return a path inside the results directory
def results_path(filename: str) -> Path:
    ensure_dir(RESULTS_DIR)
    return RESULTS_DIR / filename


# Return a path inside the notes directory
def notes_path(filename: str) -> Path:
    ensure_dir(NOTES_DIR)
    return NOTES_DIR / filename


# Return a path inside the configs directory
def config_path(filename: str) -> Path:
    ensure_dir(CONFIGS_DIR)
    return CONFIGS_DIR / filename