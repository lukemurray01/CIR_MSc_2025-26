import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.utils.io import config_path, ensure_dirs, figure_path

def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def cir_mean(
        x0: float,
        kappa: float,
        theta: float,
        t: np.ndarray,
) -> np.ndarray:
    return theta + (x0 - theta) * np.exp(-kappa * t)

def main() -> None:
    ensure_dirs()

    regimes_config = load_config("regimes.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]

    time_grid = np.linspace(0.0, 4.0, 400)

    initial_values = [0.0, theta, 4.0*theta]
    labels = [
        r"$X_0 = 0$",
        r"$X_0 = \theta$",
        r"$X_0 = 4\theta$"
    ]

    fig, ax = plt.subplots(figsize=(2.35, 1.65))

    for x0, label in zip(initial_values, labels):
        mean = cir_mean(
            x0 = x0,
            kappa = kappa,
            theta = theta,
            t = time_grid,
        )

        ax.plot(
            time_grid,
            mean,
            linewidth = 1.1,
            label = label,
        )

        ax.axhline(
            theta,
            color = "black",
            linestyle = "--",
            linewidth = 0.9,
            alpha = 0.75,
        )

        ax.text(
            time_grid[-1],
            theta,
            r"$\theta$",
            ha = "right",
            va = "bottom",
            fontsize = 7,
        )

        