import numpy as np

from experiments.run_cpu_gpu_timing import numpy_runners


def test_numpy_blt_runner_handles_non_divisible_path_batches():
    params = {"x0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.3}
    runners = numpy_runners(
        params=params,
        T=1.0,
        n_steps=8,
        h_max=1.0 / 32.0,
        seed=1234,
        klm_rho=16.0,
        blt_batch_size=5,
    )

    terminal = runners["BLT"](17)

    assert terminal.shape == (17,)
    assert np.all(np.isfinite(terminal))
    assert np.all(terminal >= 0.0)
