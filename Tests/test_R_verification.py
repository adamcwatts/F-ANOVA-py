import pandas as pd
import numpy as np
import pytest
from importlib import resources
from fanova import functionalANOVA


@pytest.fixture(scope="session")
def gait_data() -> dict[str, np.ndarray]:
    # Load packaged dataset
    with resources.files("fanova.datasets").joinpath("gait_data.csv").open("r") as f:
        df: pd.DataFrame = pd.read_csv(f)

    t_grid: np.ndarray = df["t"].to_numpy()
    group1: np.ndarray = df.filter(like="group1").to_numpy()
    group2: np.ndarray = df.filter(like="group2").to_numpy()
    group3: np.ndarray = df.filter(like="group3").to_numpy()

    return {
        "t_grid": t_grid,
        "group1": group1,
        "group2": group2,
        "group3": group3
    }


@pytest.fixture(scope="session")
def r_verif() -> dict[str, float]:
    return {
        "L2_stat": 2637.128205128205,
        "F_stat": 1.4682175466420953,
        "L2_Naive_pval": 0.2106562,
        "L2_BiasReduced_pval": 0.1957646,
        "F_Naive_pval": 0.2226683,
        "F_BiasReduced_pval": 0.2198691,
    }


class TestRVerification:

    @pytest.fixture(autouse=True)
    def _run_fanova(self, gait_data: dict[str, np.ndarray]) -> None:
        groups = [
            gait_data["group1"],
            gait_data["group2"],
            gait_data["group3"],
        ]

        self.fanova = functionalANOVA(
            groups,
            d_grid=gait_data["t_grid"],
            grid_bounds=(-np.inf, np.inf),
            group_labels=["1", "2", "3"],
        )

        self.fanova.oneway(hypothesis="FAMILY")

    def test_l2_stat(self, r_verif: dict[str, float]) -> None:
        table = self.fanova.tables.oneway
        assert table is not None
        
        l2_stat = table.loc[
            table["Family-Wise Method"] == "L2-Naive",
            "Test-Statistic",
        ].iloc[0]

        np.testing.assert_allclose(l2_stat, r_verif["L2_stat"], rtol=1e-2)

    def test_f_stat(self, r_verif: dict[str, float]) -> None:
        table = self.fanova.tables.oneway
        assert table is not None
        f_stat = table.loc[
            table["Family-Wise Method"] == "F-Naive",
            "Test-Statistic",
        ].iloc[0]

        np.testing.assert_allclose(f_stat, r_verif["F_stat"], rtol=1e-2)

    def test_l2_pvals(self, r_verif: dict[str, float]) -> None:
        table = self.fanova.tables.oneway
        assert table is not None
        for method, key in [
            ("L2-Naive", "L2_Naive_pval"),
            ("L2-BiasReduced", "L2_BiasReduced_pval"),
        ]:
            pval = table.loc[
                table["Family-Wise Method"] == method,
                "P-Value",
            ].iloc[0]

            np.testing.assert_allclose(pval, r_verif[key], rtol=1e-2)

    def test_f_pvals(self, r_verif: dict[str, float]) -> None:
        table = self.fanova.tables.oneway
        assert table is not None
        
        for method, key in [
            ("F-Naive", "F_Naive_pval"),
            ("F-BiasReduced", "F_BiasReduced_pval"),
        ]:
            pval = table.loc[
                table["Family-Wise Method"] == method,
                "P-Value",
            ].iloc[0]

            np.testing.assert_allclose(pval, r_verif[key], rtol=1e-2)