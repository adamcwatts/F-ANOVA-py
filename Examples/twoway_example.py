from scipy.io import loadmat
import os
import numpy as np
from functionalANOVA.fanova import functionalANOVA

################################################
# Import statistically significant Data (Family or Secondary Factor Only)

# Import Data
matlab_data = loadmat("Data/example_data.mat")

# Get data out of .mat file
groups = [matlab_data['TwoWayData'][0, 0], matlab_data['TwoWayData'][0, 1]]
time = matlab_data['timeData']
indicator_list = [matlab_data['IndicatorCell'][0,0], matlab_data['IndicatorCell'][0,1]]

# # Bounds on time
bounds = (-np.inf, np.inf)
n_boot = 1000 # Default it 10,000 (small for tutorial examples)
myANOVA = functionalANOVA(data_list=groups, d_grid=time, grid_bounds=bounds, subgroup_indicator=indicator_list, n_boot=n_boot)

## TwoWay Homoscedastic
myANOVA.twoway()
myANOVA.plot_means()

myANOVA.twoway( hypothesis="PRIMARY")
myANOVA.twoway( hypothesis="SECONDARY")
myANOVA.twoway( hypothesis="INTERACTION")
myANOVA.twoway( hypothesis="PAIRWISE")

myANOVA.plot_means(plot_type='PRIMARY')
myANOVA.plot_means(plot_type='Secondary')
myANOVA.plot_means(plot_type='INTERACTION')


# ## TwoWay Heteroscedastic
myANOVA.twoway_bf()
myANOVA.twoway_bf(hypothesis="PRIMARY")
myANOVA.twoway_bf(hypothesis="SECONDARY")
myANOVA.twoway_bf(hypothesis="INTERACTION")
myANOVA.twoway_bf(hypothesis="PAIRWISE")


# myANOVA.plot_covariances()


# Same Data just different indicator data type
indicator_array = matlab_data['Master_Indicator']
myANOVA = functionalANOVA(data_list=groups, d_grid=time, grid_bounds=bounds, subgroup_indicator=indicator_array)
myANOVA.twoway(n_boot=10000)
myANOVA.plot_means()


################################################
# Import non-statistically significant Data

Datamatlab_data = loadmat("Data/example_data_2.mat")
groups = [Datamatlab_data['TwoWayData'][0, 0], Datamatlab_data['TwoWayData'][0, 1]]
time = Datamatlab_data['timeData']
indicator_array = Datamatlab_data['Master_Indicator']
myANOVA_2 = functionalANOVA(data_list=groups, d_grid=time, grid_bounds=bounds, subgroup_indicator=indicator_array)
myANOVA_2.twoway(n_boot=1000)
myANOVA_2.plot_means()
