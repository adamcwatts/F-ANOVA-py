import numpy as np
from fanova import functionalANOVA
from fanova.datasets import load_example_mat
################################################
# Import statistically significant Data (Family or Secondary Factor Only)

# Load packaged .mat dataset
matlab_data = load_example_mat()

# Get data out of .mat file
groups = [matlab_data['TwoWayData'][0, 0], matlab_data['TwoWayData'][0, 1]]
time = matlab_data['timeData']
indicator_list = [matlab_data['IndicatorCell'][0,0], matlab_data['IndicatorCell'][0,1]]

# # Bounds on time
bounds = (-np.inf, np.inf)

myANOVA = functionalANOVA(data_list=groups, d_grid=time, grid_bounds=bounds, subgroup_indicator=indicator_list,
                          primary_labels=["Bladder", "Bottle"],
                          secondary_labels=["Pants", "Shorts", "Skirt", "Dress"])

myANOVA.twoway(hypothesis="custom", methods = ["L2-Simul", 'F-Simul'],  contrast = [-1, 1])

# myANOVA.twoway(hypothesis="custom",  contrast = [0, -1, 0, 1])

