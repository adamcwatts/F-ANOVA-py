from scipy.io import loadmat
import os
import numpy as np
from functionalANOVA.core.fanova import functionalANOVA

################################################
# Import statistically significant Data (Family or Secondary Factor Only)

# Import Data
script_dir = os.path.dirname(__file__)  # folder containing the script
matlab_data = loadmat(os.path.join(script_dir,'Data', "example_data.mat"))

# Get data out of .mat file
groups = [matlab_data['TwoWayData'][0, 0], matlab_data['TwoWayData'][0, 1]]
time = matlab_data['timeData']
indicator_list = [matlab_data['IndicatorCell'][0,0], matlab_data['IndicatorCell'][0,1]]

# # Bounds on time
bounds = (-np.inf, np.inf)

myANOVA = functionalANOVA(data_list=groups, d_grid=time, grid_bounds=bounds, subgroup_indicator=indicator_list)

## TwoWay Homoscedastic
myANOVA.twoway(n_boot=1000)
myANOVA.twoway(n_boot=1000, hypothesis="PRIMARY")
myANOVA.twoway(n_boot=1000, hypothesis="SECONDARY")
myANOVA.twoway(n_boot=1000, hypothesis="INTERACTION")
myANOVA.twoway(n_boot=1000, hypothesis="PAIRWISE")

myANOVA.plot_means(plot_type='PRIMARY')
myANOVA.plot_means(plot_type='Secondary')
myANOVA.plot_means(plot_type='INTERACTION')


# ## TwoWay Heteroscedastic
myANOVA.twoway_bf(n_boot=1000)
myANOVA.twoway_bf(n_boot=1000, hypothesis="PRIMARY")
myANOVA.twoway_bf(n_boot=1000, hypothesis="SECONDARY")
myANOVA.twoway_bf(n_boot=1000, hypothesis="INTERACTION")
myANOVA.twoway_bf(n_boot=1000, hypothesis="PAIRWISE")


# myANOVA.plot_covariances()


# Same Data just different indicator data type
indicator_array = matlab_data['Master_Indicator']
myANOVA = functionalANOVA(data_list=groups, d_grid=time, grid_bounds=bounds, subgroup_indicator=indicator_array)
myANOVA.twoway(n_boot=10000)
myANOVA.plot_means()


################################################
# Import non-statistically significant Data

Datamatlab_data = loadmat(os.path.join(script_dir,'Data', "example_data_2.mat"))
groups = [Datamatlab_data['TwoWayData'][0, 0], Datamatlab_data['TwoWayData'][0, 1]]
time = Datamatlab_data['timeData']
indicator_array = Datamatlab_data['Master_Indicator']
myANOVA = functionalANOVA(data_list=groups, d_grid=time, grid_bounds=bounds, subgroup_indicator=indicator_array)
myANOVA.twoway(n_boot=1000)
myANOVA.plot_means()