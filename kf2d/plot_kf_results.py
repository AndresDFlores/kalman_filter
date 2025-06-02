import numpy as np
from matplotlib import pyplot as plt


class PlotKF:

    def __init__(self, save_figs=False):
        self.save_figs = save_figs


    
    def get_error(self, results):

        #  calculate error between measured and predicted
        error = np.array(results['measured']) - np.array(results['predicted'])

        #  mean square error:  average of the squared differences between predicted and actual values
        self.mse = np.sum(error**2)/len(error)

        #  root mean square:  average error value in the same units as the target variable
        self.rms = np.sqrt(self.mse)



    def plot_results(self, results, kf_dim):
        
        #  init new plot
        self.fig, self.ax = plt.subplots(figsize=(7, 5))


        #  plot measured
        x = results['idx']
        y = results['measured']
        plt.plot(x, y, color='green', label='measured')


        #  plot predicted
        x = results['idx']
        y = results['predicted']
        plt.plot(x, y, ls='--', color='black', label='predicted')


        #  plot kalman
        x = results['idx']
        y = results['kalman']
        plt.plot(x, y, ls='--', color='red', label='kalman')


        #  calculate error between predicted and measured
        self.get_error(results)


        #  format plot
        self.ax.set_title(f'Kalman Filter RMS: {round(self.rms, 3)}')
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel(f'{kf_dim}')
        self.ax.grid(True)
        self.ax.legend()


        #  save figure as .png
        if self.save_figs: plt.savefig(f'KalmanFilter_{kf_dim}.png', dpi=300)

