from importlib import resources
from scipy.io import loadmat
import pandas as pd


def load_example_csv() -> pd.DataFrame:
    with resources.files("fanova.datasets").joinpath("gait_data.csv").open("r") as f:
        df: pd.DataFrame = pd.read_csv(f)
        return df

def load_example_mat():
    with resources.files("fanova.datasets").joinpath("example_data.mat").open("rb") as f:
        return loadmat(f)

def load_example_mat2():
    with resources.files("fanova.datasets").joinpath("example_data_2.mat").open("rb") as f:
        return loadmat(f)