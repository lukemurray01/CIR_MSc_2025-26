import numpy as np
import yaml

# Pull functions needed for testing
from src.utils.io import config_path
from src.utils.cir_params import cir_delta, kl_alpha

# Load the regimes.yaml file for testing purposes
def _load_config() -> dict:
    with open(config_path("regimes.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)

# Ensure that the parameters within the regimes.yaml are correctly synced between files
def test_config_has_shared_block():
    config = _load_config()
    assert config["shared"]["kappa"] == 2.0
    assert config["shared"]["theta"] == 0.02
    assert config["shared"]["x0"] == 0.02

# Ensure that the yaml file indeed contains the five regimes covered for the thesis
def test_config_contains_all_five_regimes():
    config = _load_config()
    keys = set(config["regimes"].keys())
    assert keys == {"A", "B", "C", "D", "E"}

# Ensure that the regime deltas match within machine preicison (tolerance of 0.001)
def test_regime_deltas_match_definitions():
    config = _load_config()
    kappa, theta = config["shared"]["kappa"], config["shared"]["theta"]
    expected = {"A": 16.0, "B": 4.0, "C": 2.0, "D": 0.64, "E": 0.25}

    for name, expected_delta in expected.items():

        sigma = config["regimes"][name]["sigma"]

        delta = cir_delta(kappa, theta, sigma)

        np.testing.assert_allclose(delta, expected_delta, rtol=0.001,
                                    err_msg=f"Regime {name}")

# Ensure that the alpha parameter for the Kelly-Lord changes sign for Feller violated regimes
def test_kl_alpha_sign_matches_feller_status():
    config = _load_config()

    kappa, theta = config["shared"]["kappa"], config["shared"]["theta"]

    expected_sign = {"A": 1, "B": 1, "C": 1, "D": -1, "E": -1} # "Regime name"- "sign"

    for name, sign in expected_sign.items():
        sigma = config["regimes"][name]["sigma"]
        alpha = kl_alpha(kappa, theta, sigma)
        assert np.sign(alpha) == sign, f"Regime {name}: alpha={alpha}"