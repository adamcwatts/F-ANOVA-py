import numpy as np
import pandas as pd
from fanova import functionalANOVA
from fanova.datasets import load_example_csv

# Import Data
df = load_example_csv()
# Extract all group columns based on column name patterns
group1_cols = [col for col in df.columns if col.startswith("group1")]
group2_cols = [col for col in df.columns if col.startswith("group2")]
group3_cols = [col for col in df.columns if col.startswith("group3")]

# Convert to NumPy arrays
group1 = df[group1_cols].to_numpy()
group2 = df[group2_cols].to_numpy()
group3 = df[group3_cols].to_numpy()

# Final structure
group_arrays = [group1, group2, group3]

# Extract time vector
time = df["t"].to_numpy()

# Bounds on time
bounds = (-np.inf, np.inf)
n_boot = 1000 # Default it 10,000 (small for tutorial examples)
myANOVA = functionalANOVA(data_list=group_arrays, d_grid=time, grid_bounds=bounds,
                          group_labels=['Group A', 'Group B', 'Group C'],
                          domain_label='Time', domain_units='Seconds',
                          response_label='Temperature', response_units='Celcius',
                          n_boot=n_boot)

myANOVA.oneway(hypothesis='family')
myANOVA.oneway(hypothesis='pairwise')

myANOVA.oneway_bf(hypothesis='family')
myANOVA.oneway_bf(hypothesis='pairwise')

myANOVA.plot_means()
myANOVA.plot_covariances()

