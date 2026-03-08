import numpy as np
import pytest
from typing import Callable
from dataclasses import dataclass, asdict
from fanova import fanova
from utils import *

@dataclass
class LogisticParams:
    L: float = 7
    k: float = 0.75
    x_0: float = 10

# contains just length values for rbf kernel
@dataclass
class RBFParams:
    l1: float = 0.25
    l2: float = 2.5
    l3: float = 25


@pytest.fixture(scope="session")
def log_params() -> LogisticParams:
    return LogisticParams()

@pytest.fixture(scope="session")
def rbf_params() -> RBFParams:
    return RBFParams()

@pytest.fixture(scope="session")
def logistic_init(log_params: LogisticParams, rbf_params: RBFParams) -> dict:
    t_grid = np.arange(0,25,0.01)
    eta_t = logistic_function(log_params.L, log_params.k, log_params.x_0, t_grid)
    l_values: np.ndarray = np.array(list(asdict(rbf_params).values()))

    cov_matricies: list[np.ndarray] = []
    for l in l_values:
        kernel = lambda s,t: rbf_kernel(s,t,l)
        cov_matricies.append(covariance_matrix(kernel, t_grid))

    return {
        "t_grid": t_grid,
        "eta_t": eta_t,
        "cov_matricies": cov_matricies,
    }


class TestOneway:
    @pytest.mark.parametrize("cov_idx",[
        pytest.param(0,id="l=0.25"),
        pytest.param(1,id="l=2.5"),
        pytest.param(2,id="l=25"),
    ])
    def test_equal_means(self, logistic_init: dict,cov_idx: int) -> None:
        rng: np.random.Generator = init_rng(9000)
        n_reals: int = 25
        n_groups: int = 3

        cov: np.ndarray = logistic_init["cov_matricies"][cov_idx]

        groups: list[np.ndarray] = []
        for k in range(n_groups):
            data = generate_logistic_data(logistic_init["eta_t"], cov, n_reals, rng)
            groups.append(data)

        fanova: functionalANOVA = functionalANOVA(
            groups,
            d_grid=logistic_init["t_grid"],
            grid_bounds=(-np.inf,np.inf)
        )

        fanova.oneway(
            hypothesis="FAMILY",
            seed=9000
        )

        p_vals = fanova.tables.oneway["P-Value"].values
        assert np.all(p_vals > 0.05)

    @pytest.mark.parametrize("hypothesis_type", [
        "FAMILY",
        "PAIRWISE"
    ])
    def test_shifted_means(self, logistic_init: dict, hypothesis_type: str) -> None:
        rng: np.random.Generator = init_rng(9001)
        n_reals: int = 15
        n_groups: int = 3

        cov: np.ndarray = logistic_init["cov_matricies"][1]

        groups: list[np.ndarray] = []

        for g in range(n_groups):
            data: np.ndarray = generate_logistic_data(logistic_init["eta_t"], cov, n_reals, rng)
            if g == 0:
                data += rng.integers(1,6)
            groups.append(data)

        fanova: functionalANOVA = functionalANOVA(
            groups,
            d_grid=logistic_init["t_grid"],
            grid_bounds=(-np.inf,np.inf)
        )

        fanova.oneway(
            hypothesis=hypothesis_type,
            seed=9001
        )

        if hypothesis_type == "FAMILY":
            p_vals = fanova.tables.oneway["P-Value"].values
            assert np.all(p_vals < 0.05)
        else: # pairwise
            assert np.all(fanova.tables.oneway.iloc[0,1:] < 0.05)  # 0 & 1
            assert np.all(fanova.tables.oneway.iloc[1,1:] < 0.05)  # 0 & 2
            assert np.all(fanova.tables.oneway.iloc[2,1:] > 0.05)  # 1 & 2

class TestOnewayBF:
    def test_equal_means(self, logistic_init: dict) -> None:
        rng: np.random.Generator = init_rng(9001)
        n_reals: int = 30
        n_groups: int = 3

        groups: list[np.ndarray] = []
        for k in range(n_groups):
            data = generate_logistic_data(logistic_init["eta_t"], logistic_init["cov_matricies"][k], n_reals, rng)
            groups.append(data)

        fanova: functionalANOVA = functionalANOVA(
            groups,
            d_grid=logistic_init["t_grid"],
            grid_bounds=(-np.inf, np.inf)
        )

        fanova.oneway_bf(
            hypothesis="FAMILY",
            seed=9001
        )

        p_vals = fanova.tables.oneway_bf["P-Value"].values
        assert np.all(p_vals > 0.05)

    @pytest.mark.parametrize("hypothesis_type", [
        "FAMILY",
        "PAIRWISE"
    ])
    def test_shifted_means(self, logistic_init: dict, hypothesis_type: str) -> None:
        rng: np.random.Generator = init_rng(9001)
        n_reals: int = 30
        n_groups: int = 3

        groups: list[np.ndarray] = []
        for k in range(n_groups):
            data = generate_logistic_data(logistic_init["eta_t"], logistic_init["cov_matricies"][k], n_reals, rng)
            if k == 2:
                data += 1
            groups.append(data)

        fanova: functionalANOVA = functionalANOVA(
            groups,
            d_grid=logistic_init["t_grid"],
            grid_bounds=(-np.inf, np.inf)
        )

        fanova.oneway_bf(
            hypothesis=hypothesis_type,
            seed=9001
        )

        if hypothesis_type == "FAMILY":
            p_vals = fanova.tables.oneway_bf["P-Value"].values
            assert np.all(p_vals < 0.05)
        else: # pairwise
            table = fanova.tables.oneway_bf
            assert np.all(table.iloc[0,1:] > 0.05)  # 0 & 1
            assert np.all(table.iloc[1,1:] < 0.05)  # 0 & 2
            assert np.all(table.iloc[2,1:] < 0.05)  # 1 & 2
