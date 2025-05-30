from operator import itemgetter

import pandas as pd

from kalman1d import *
from kalman_filter import *

from plot_kf_results import *


#  load data from source file
data_files = ['demo_point_trajectories.xlsx']
file = data_files[0]

data = pd.read_excel(file, sheet_name='obj_0')
rows, cols = data.shape


#  convert loaded dataframe into a dictionary
data_dict = data.to_dict('list')
data_lists = [val for _,val in data_dict.items()]


#  init kalman filter
kf_class = Kalman1D()


#  initialize kf variables
kf_class.set_dt(1)  # set time delta between data points
kf_class.set_initial_process_covariance_matrix()  # initialize process covariance matrix



#  iterate through sample data to simulate data acquisition
for idx, row in enumerate(range(rows)):

    #  set iteration index
    kf_class.set_iter(idx)
    print('\n')
    print(kf_class.iter)


    #  measure current position data
    time_curr, x_pos_curr, x_vel_curr = list(map(itemgetter(idx), data_lists))


    kf_class.measured['time'].append(time_curr)
    kf_class.measured['x_pos'].append(x_pos_curr)
    kf_class.measured['x_vel'].append(x_vel_curr)


    #  update measured state matrix values with current measured values
    current_state_dict = dict(
        x_pos = kf_class.measured['x_pos'][-1],
        x_vel = kf_class.measured['x_vel'][-1],
    )

    kf_class.set_current_state_measurement(current_state_dict)


    #  kalman filter implementation after 10 data acquisition cycles
    if idx<1:
        kf_class.set_initial_state_matrix()

    else:
        kf_class.main()



#  plot kalman filter performance results
plot_kf_class = PlotKF(save_figs=True)
for dim in kf_class.kf_results.keys():
    plot_kf_class.plot_results(kf_class.kf_results[dim], kf_dim=dim)