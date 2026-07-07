from importlib import import_module

import yaml

from src.utils.io import config_path
from src.utils.style import METHOD_COLOURS, METHOD_LABELS, METHOD_LINESTYLES


# Takes a YAML filename and returns its contents as a Python dictionary.
def _load_yaml(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_experiments_block_is_nonempty():
    config = _load_yaml("experiments.yaml")

    assert "experiments" in config
    assert isinstance(config["experiments"], dict)
    assert len(config["experiments"]) > 0

def test_each_experiment_has_required_fields():
    config = _load_yaml("experiments.yaml")

    for name, experiment in config["experiments"].items():
        assert experiment is not None, f"Experiment {name} is empty or incorrectly indented"
        assert "description" in experiment, f"Experiment {name} is missing description"
        assert "regimes" in experiment, f"Experiment {name} is missing regimes"
        assert "T" in experiment, f"Experiment {name} is missing T"
        assert "n_paths" in experiment, f"Experiment {name} is missing n_paths"

# Checks that every regime used by an experiment is defined in regimes.yaml.
def test_all_experiment_regimes_exist_in_regimes_config():
    experiments_config = _load_yaml("experiments.yaml")
    regimes_config = _load_yaml("regimes.yaml")

    valid_regimes = set(regimes_config["regimes"].keys())

    for name, experiment in experiments_config["experiments"].items():
        regimes = set(experiment["regimes"])
        assert regimes.issubset(valid_regimes), name


def test_method_groups_match_thesis_roles():
    config = _load_yaml("experiments.yaml")

    core = set(config["methods"]["core"])
    reference_only = set(config["methods"]["reference_only"])
    terminal_only = set(config["methods"]["terminal_only"])

    assert core == {"FTE", "ProjEuler", "KL", "KLM", "BLT"}
    assert reference_only == {"HH"}
    assert terminal_only == {"Exact"}
    assert core.isdisjoint(reference_only)
    assert core.isdisjoint(terminal_only)
    assert reference_only.isdisjoint(terminal_only)


# Checks that terminal-law methods are not accidentally treated as core path methods.
def test_terminal_only_methods_are_not_core_methods():
    config = _load_yaml("experiments.yaml")

    core = set(config["methods"]["core"])
    terminal_only = set(config["methods"]["terminal_only"])

    assert "Exact" in terminal_only
    assert "Exact" not in core


def test_registered_methods_have_styles_and_labels():
    config = _load_yaml("experiments.yaml")

    methods = set()
    for group in config["methods"].values():
        methods.update(group)

    assert methods.issubset(METHOD_COLOURS)
    assert methods.issubset(METHOD_LINESTYLES)
    assert methods.issubset(METHOD_LABELS)


def test_registered_methods_have_importable_implementations():
    config = _load_yaml("experiments.yaml")

    registry = {
        "FTE": ("src.samplers.full_truncation_euler", "fte_terminal_from_dW"),
        "ProjEuler": (
            "src.samplers.projected_euler",
            "projected_euler_terminal_from_dW",
        ),
        "KL": ("src.samplers.kelly_lord", "kl_uniform_terminal_from_dW"),
        "KLM": ("src.samplers.klm_backstop", "klm_backstop_terminal_from_fine_dW"),
        "BLT": ("src.samplers.blt_splitting", "blt_terminal_from_noise"),
        "HH": ("src.samplers.hh_milstein", "hh_milstein_terminal_from_dW"),
        "Exact": ("src.samplers.exact", "cir_ncx2_params"),
    }

    methods = set()
    for group in config["methods"].values():
        methods.update(group)

    assert methods == set(registry)
    for method in sorted(methods):
        module_name, attr_name = registry[method]
        module = import_module(module_name)
        assert hasattr(module, attr_name), method


# Checks that the strong-error reference grid is finer than every coarse grid.
def test_strong_error_reference_grid_is_finer_than_coarse_grids():
    config = _load_yaml("experiments.yaml")

    strong_error = config["time_grids"]["strong_error"]

    assert strong_error["reference_n_steps"] > max(strong_error["coarse_n_steps"])


def test_production_reference_resolution_is_thesis_safe_default():
    config = _load_yaml("experiments.yaml")

    assert config["time_grids"]["strong_error"]["reference_n_steps"] >= 16384
    assert config["klm_fig3_jax"]["full"]["reference_power"] >= 16


def test_reference_sensitivity_ladders_cover_current_production_defaults():
    config = _load_yaml("experiments.yaml")

    strong = config["reference_sensitivity"]["strong_error"]
    fig3 = config["reference_sensitivity"]["klm_fig3_jax"]

    assert strong["reference_n_steps"] == [4096, 16384, 32768]
    assert config["time_grids"]["strong_error"]["reference_n_steps"] in strong[
        "reference_n_steps"
    ]
    assert fig3["reference_powers"] == [14, 16, 18]
    assert config["klm_fig3_jax"]["full"]["reference_power"] in fig3[
        "reference_powers"
    ]
