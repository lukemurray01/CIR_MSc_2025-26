"""Validation of the analytical control-variate derivatives.

Every d_x u is checked against a central finite difference of an
independently computed value function (quadrature for g2, the existing
closed forms for g1/g3/g4), and the mean-zero property of the assembled
control is checked directly by simulation.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import ncx2

from src.metrics.control_variate import (
    G2Table,
    _G_squared_call,
    cir_bond_AB,
    control_from_paths,
    dxu_g1,
    dxu_g2_exact,
    dxu_g3,
    dxu_g4,
    running_trapezoid,
    u_g3,
)
from src.metrics.weak_error import (
    affine_cir_bond_price,
    exact_cir_mean,
    trapezoidal_integral,
)
from src.samplers.exact import cir_ncx2_params
from src.samplers.full_truncation_euler import fte_paths_from_dW

KAPPA, THETA, STRIKE = 2.0, 0.02, 0.02
REGIME_SIGMA = {"A": 0.10, "C": 0.282842712474619, "E": 0.80}


def u2_quadrature(tau, x, kappa, theta, sigma, strike=STRIKE):
    """E[(X_T - K)_+^2 | X_t = x] by direct quadrature (reference)."""
    c, df, nc = cir_ncx2_params(x=x, kappa=kappa, theta=theta, sigma=sigma, dt=tau)
    val, _ = quad(
        lambda y: (y - strike) ** 2 * c * ncx2.pdf(c * y, df=df, nc=nc),
        strike, np.inf, epsabs=1e-14, epsrel=1e-12, limit=300,
    )
    return val


# ------------------------------------------------------------------ g2 -----

@pytest.mark.parametrize("regime", ["A", "C", "E"])
@pytest.mark.parametrize("tau", [1.0, 0.25, 0.01])
def test_G_squared_call_matches_quadrature(regime, tau):
    """The truncated-moment identities reproduce E[(Z - z)_+^2]."""
    sigma = REGIME_SIGMA[regime]
    for x in (0.005, 0.02, 0.05):
        c, df, nc = cir_ncx2_params(x=x, kappa=KAPPA, theta=THETA,
                                    sigma=sigma, dt=tau)
        z = c * STRIKE
        got = _G_squared_call(z, df, nc)
        want, _ = quad(
            lambda y: (y - z) ** 2 * ncx2.pdf(y, df=df, nc=nc),
            z, np.inf, epsabs=1e-14, epsrel=1e-12, limit=300,
        )
        assert got == pytest.approx(want, rel=1e-7, abs=1e-14)


@pytest.mark.parametrize("regime", ["A", "C", "E"])
@pytest.mark.parametrize("tau", [1.0, 0.25, 0.01])
def test_dxu_g2_matches_finite_difference(regime, tau):
    """The lambda-derivative identity reproduces d_x u2."""
    sigma = REGIME_SIGMA[regime]
    for x in (0.008, 0.02, 0.045):
        eps = 1e-6 * max(x, 1e-3)
        fd = (
            u2_quadrature(tau, x + eps, KAPPA, THETA, sigma)
            - u2_quadrature(tau, x - eps, KAPPA, THETA, sigma)
        ) / (2 * eps)
        got = float(dxu_g2_exact(tau, x, KAPPA, THETA, sigma))
        assert got == pytest.approx(fd, rel=2e-4, abs=1e-12)


@pytest.mark.parametrize("regime", ["A", "E"])
def test_g2_table_interpolation_is_accurate(regime):
    """Tabulated d_x u2 tracks the exact value closely enough to hedge."""
    sigma = REGIME_SIGMA[regime]
    table = G2Table(KAPPA, THETA, sigma, T=1.0, tau_min=2.0**-12, strike=STRIKE)
    rng = np.random.default_rng(0)
    taus = np.exp(rng.uniform(np.log(2.0**-12), 0.0, size=40))
    xs = rng.uniform(0.0, 0.08, size=40)
    got = np.array([table(t, np.array([x]))[0] for t, x in zip(taus, xs)])
    want = np.array([float(dxu_g2_exact(t, x, KAPPA, THETA, sigma))
                     for t, x in zip(taus, xs)])
    scale = max(np.max(np.abs(want)), 1e-12)
    assert np.max(np.abs(got - want)) / scale < 5e-3


# -------------------------------------------------------------- g1/g3/g4 ---

def test_dxu_g1_matches_finite_difference():
    for tau in (1.0, 0.3, 0.02):
        fd = (
            exact_cir_mean(0.02 + 1e-7, KAPPA, THETA, tau)
            - exact_cir_mean(0.02 - 1e-7, KAPPA, THETA, tau)
        ) / 2e-7
        assert float(dxu_g1(tau, KAPPA)) == pytest.approx(fd, rel=1e-6)


@pytest.mark.parametrize("regime", ["A", "C", "E"])
def test_dxu_g3_matches_finite_difference(regime):
    sigma = REGIME_SIGMA[regime]
    for tau in (1.0, 0.3, 0.02):
        for x in (0.005, 0.02, 0.05):
            eps = 1e-7
            fd = (
                u_g3(tau, x + eps, KAPPA, THETA, sigma)
                - u_g3(tau, x - eps, KAPPA, THETA, sigma)
            ) / (2 * eps)
            got = float(dxu_g3(tau, x, KAPPA, THETA, sigma))
            assert got == pytest.approx(fd, rel=1e-5)


@pytest.mark.parametrize("regime", ["A", "C", "E"])
def test_bond_AB_reproduces_existing_closed_form(regime):
    """cir_bond_AB must agree with the shipped affine_cir_bond_price."""
    sigma = REGIME_SIGMA[regime]
    for tau in (1.0, 0.3, 0.02):
        for x in (0.0, 0.02, 0.05):
            A, B = cir_bond_AB(tau, KAPPA, THETA, sigma)
            got = float(A * np.exp(-B * x))
            want = affine_cir_bond_price(x0=x, kappa=KAPPA, theta=THETA,
                                         sigma=sigma, T=tau)
            assert got == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("regime", ["A", "E"])
def test_dxu_g4_matches_finite_difference(regime):
    sigma = REGIME_SIGMA[regime]
    for tau in (1.0, 0.3, 0.02):
        for x in (0.005, 0.02, 0.05):
            eps = 1e-7
            fd = (
                affine_cir_bond_price(x + eps, KAPPA, THETA, sigma, tau)
                - affine_cir_bond_price(x - eps, KAPPA, THETA, sigma, tau)
            ) / (2 * eps)
            got = float(dxu_g4(tau, x, 1.0, KAPPA, THETA, sigma))
            assert got == pytest.approx(fd, rel=1e-5)


def test_running_trapezoid_endpoint_matches_shipped_integral():
    rng = np.random.default_rng(3)
    paths = np.abs(rng.normal(0.02, 0.01, size=(50, 65)))
    dt = 1.0 / 64
    assert np.allclose(
        running_trapezoid(paths, dt)[:, -1], trapezoidal_integral(paths, dt),
        rtol=1e-13, atol=0.0,
    )


# --------------------------------------------------------- mean-zero -------

@pytest.mark.parametrize("payoff", ["g1", "g2", "g3", "g4"])
def test_control_is_mean_zero(payoff):
    """E[C_h] = 0 to Monte Carlo accuracy -- the property that makes the
    control unable to bias the weak-error estimate."""
    sigma, T, n_steps, n_paths = REGIME_SIGMA["E"], 1.0, 64, 40_000
    dt = T / n_steps
    rng = np.random.default_rng(11)
    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    paths = fte_paths_from_dW(X0=0.02, kappa=KAPPA, theta=THETA, sigma=sigma,
                              dt=dt, dW=dW)

    g2_rows = None
    if payoff == "g2":
        table = G2Table(KAPPA, THETA, sigma, T=T, tau_min=dt, strike=STRIKE)
        taus = T - dt * np.arange(n_steps)
        g2_rows = {"x": table.x, "rows": table.rows_for_grid(taus)}

    C = control_from_paths(payoff, paths, dW, dt, T, KAPPA, THETA, sigma,
                           g2_rows=g2_rows)
    se = C.std(ddof=1) / np.sqrt(n_paths)
    assert abs(C.mean()) < 4.0 * se


# ------------------------------------------------- audit-driven checks -----

@pytest.mark.parametrize("regime", ["A", "C", "E"])
def test_exact_g2_reference_matches_quadrature(regime):
    """The closed-form g2 reference agrees with the quadrature it replaced."""
    from src.metrics.weak_error import (
        exact_g2_squared_call_by_quadrature,
        exact_g2_squared_call_expectation,
    )

    sigma = REGIME_SIGMA[regime]
    kw = dict(x0=0.02, kappa=KAPPA, theta=THETA, sigma=sigma, T=1.0,
              strike=STRIKE)
    closed = exact_g2_squared_call_expectation(**kw)
    quadrature = exact_g2_squared_call_by_quadrature(**kw)
    # Far below the ~1e-6 biases the deep ladder reports.
    assert abs(closed - quadrature) < 1e-15


def _trapezoid_bond_on_exact_skeleton(sigma, n_steps, kappa=KAPPA,
                                      theta=THETA, x0=0.02, T=1.0):
    """E[exp(-trapezoid(X))] on an EXACT CIR skeleton, no Monte Carlo.

    Backward affine recursion: the value function stays exponential-affine,
    exp(-a - b x), because each CIR transition has an affine Laplace
    exponent.  This isolates the trapezoidal rule's own error from any
    scheme error.
    """
    h = T / n_steps
    delta = 4.0 * kappa * theta / sigma**2
    q = sigma**2 * (1.0 - np.exp(-kappa * h)) / (2.0 * kappa)
    decay = np.exp(-kappa * h)

    def A(u):
        return (delta / 2.0) * np.log(1.0 + u * q)

    def B(u):
        return u * decay / (1.0 + u * q)

    a, b = 0.0, h / 2.0                       # weight 1/2 on X_T
    for _ in range(n_steps - 1, 0, -1):       # weight 1 on interior nodes
        a, b = a + A(b), h + B(b)
    a, b = a + A(b), h / 2.0 + B(b)           # weight 1/2 on X_0
    return float(np.exp(-a - b * x0))


@pytest.mark.parametrize("regime", ["A", "C", "E"])
def test_g4_trapezoid_error_is_second_order(regime):
    """The g4 path-integral quadrature converges at order 2, not order 1.

    If it were order 1 it would sit at the same rate as the schemes' weak
    bias and the g4 column would partly measure the quadrature rule rather
    than the scheme.  It does not: measured local order is 2.00, and in the
    worst regime the term is 7e-12 at h = 2^-12, far below any bias the
    experiment reports.
    """
    sigma = REGIME_SIGMA[regime]
    exact = affine_cir_bond_price(x0=0.02, kappa=KAPPA, theta=THETA,
                                  sigma=sigma, T=1.0)
    errors = {
        k: abs(_trapezoid_bond_on_exact_skeleton(sigma, 2**k) - exact)
        for k in (6, 9, 12)
    }
    order = np.log2(errors[6] / errors[9]) / 3.0
    assert 1.9 < order < 2.1, f"{regime}: order {order:.3f}, {errors}"
    assert errors[12] < 1e-10
