from operator import itemgetter

import pandas as pd

from kalman2d import *
from kalman_filter import *

from plot_kf_results import *


#  display messages
literal = True


#  how many iterations to pass before starting the kalman filter
kf_iter_start = 5


#  load data from source file
data_files = ['demo_data.xlsx']
file = data_files[0]

data = pd.read_excel(file, sheet_name='obj_0').head(15)
rows, cols = data.shape


#  convert loaded dataframe into a dictionary
data_dict = data.to_dict('list')
data_lists = [val for _,val in data_dict.items()]


#  init kalman filter
kf_class = Kalman2D(literal)



#  iterate through sample data to simulate data acquisition
for idx, row in enumerate(range(rows)):

    #  set iteration index
    kf_class.set_iter(idx)
    if literal: print('\n---> ITERATION: ', kf_class.iter)


    #  measure current position data
    time_curr, x_pos_curr, y_pos_curr = list(map(itemgetter(idx), data_lists))


    kf_class.measured['time'].append(time_curr)
    kf_class.measured['x_pos'].append(x_pos_curr)
    kf_class.measured['y_pos'].append(y_pos_curr)


    #  calculate velocity from positional data
    if idx>=1:
        dt = kf_class.measured['time'][-1] - kf_class.measured['time'][-2]
        x_vel_curr = (kf_class.measured['x_pos'][-1] - kf_class.measured['x_pos'][-2])/dt
        y_vel_curr = (kf_class.measured['y_pos'][-1] - kf_class.measured['y_pos'][-2])/dt

    else:
        dt = 0
        x_vel_curr = 0
        y_vel_curr = 0


    kf_class.measured['x_vel'].append(x_vel_curr)
    kf_class.measured['y_vel'].append(y_vel_curr)


    #  update measured state matrix values with current measured values
    current_state_dict = dict(
        x_pos = kf_class.measured['x_pos'][-1],
        y_pos = kf_class.measured['y_pos'][-1],
        x_vel = kf_class.measured['x_vel'][-1],
        y_vel = kf_class.measured['y_vel'][-1],
    )


    #  set current state measurement
    kf_class.set_current_state_measurement(current_state_dict)


    #  initialize kf process covariance matrix
    if idx==kf_iter_start-1:
        kf_class.set_initial_process_covariance_matrix(dim_keys=kf_class.dim_keys[1:])


    #  kalman filter implementation after 10 data acquisition cycles
    if idx<kf_iter_start:
        kf_class.set_initial_state_matrix()

    else:
        kf_class.set_dt(dt)  # set time delta between data points
        kf_class.main()



#  plot kalman filter performance results
plot_kf_class = PlotKF(save_figs=True)
for dim in kf_class.kf_results.keys():
    if literal: plot_kf_class.plot_results(kf_class.kf_results[dim], kf_dim=dim)