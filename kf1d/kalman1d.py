import numpy as np
from kalman_filter import *


class Kalman1D(KalmanFilter):


    def __init__(self, literal=False):

        # child inherits all methods/props from parent
        # do not need name of parent element, automatically inherit meths/props from parent
        super().__init__()

        #  display messages
        self.literal = literal

        #  log all measured data
        self.dim_keys = ['time', 'x_pos', 'x_vel']
        self.measured = {key:[] for key in self.dim_keys}  # replace lists with dequeue to keep this from growing indefinitely

        #  record predictions and measurements by iter index
        self.kf_results = dict(
            x_pos = dict(
                idx=[],
                predicted=[],
                measured=[],
                kalman=[]
            ),
            x_vel = dict(
                idx=[],
                predicted=[],
                measured=[],
                kalman=[]
            ),
        )



    @classmethod
    def set_iter(cls, iter_idx):
        cls.iter = iter_idx
        


    @classmethod
    def set_dt(cls, dt):
        cls.dt = dt



    @classmethod
    def set_current_state_measurement(cls, current_state_dict):

        current_state = np.matrix(
            [
                [current_state_dict['x_pos']],
                [current_state_dict['x_vel']],

            ]
        )

        cls.current_state = current_state



    def set_initial_state_matrix(self):
        self.x_prev = self.current_state



    def set_initial_process_covariance_matrix(self):

        covariance_matrix = np.matrix(
            [
                [400, 0],
                [0, 25]
            ]
        )

        self.P_prev = covariance_matrix



    def set_state_matrices(self):

        A = np.matrix(
            [
                [1, self.dt],
                [0, 1],
            ]
        )


        B = np.matrix(
            [
                [0.5*(self.dt**2)],
                [self.dt],
            ]
        )



        u = np.matrix(
            [
                [2]
            ]
        )

        return A, B, u
    


    def set_measured_input_matrices(self):

        C = np.matrix(
            [
                [1, 0],
                [0, 1],
            ]
        )

        z = 0

        return C, z



    def set_process_covariance_matrix(self, A, P_prev, Q):

        process_covariance_matrix = A@P_prev@np.transpose(A)
        
        return process_covariance_matrix



    def set_sensor_noise_covariance_matrix(self, observation_error_matrix):

        #  defines how the state variables are transformed into measurements
        sensor_noise_covariance_matrix = observation_error_matrix**2

        return sensor_noise_covariance_matrix
    


    def set_H(self, P):

        #  identity matrix
        H = np.identity(P.shape[0])

        return H
    


    def main(self):


        #  ----- PREDICT

        #  predict next state
        A, B, u = self.set_state_matrices()
        x_curr_pred = self.predict_next_state(A, self.x_prev, B, u)
        if self.literal: print(f'\nX_CURR_PRED:\n{x_curr_pred}')


        #  predict process Covariance Matrix
        P_pred = self.predict_process_covariance(A, self.P_prev)
        if self.literal: print(f'\nP_PRED:\n{P_pred}')



        #  ----- MEASURE

        #  measured data
        C, z = self.set_measured_input_matrices()
        Y = self.get_measured_input(C, self.current_state, z)
        if self.literal: print(f'\nY:\n{Y}')



        #  ----- CORRECT

        #  observation error (DOES NOT update) - tune this to influence KF performance
        observation_error_matrix = np.matrix(
            [
                [25, 0],
                [0, 6]
            ]
        )  # [[22.5, 0], [0, 1]], calculate variance for each axis as a starting point


        #  sensor noise covariance matrix
        R = self.set_sensor_noise_covariance_matrix(observation_error_matrix)
        if self.literal: print(f'\nR:\n{R}')


        #  adaptation matrix
        H = self.set_H(self.P_prev)
        if self.literal: print(f'\nH:\n{H}')


        #  calculate kalman gain
        K = self.get_kalman_gain(P_pred, H, R)
        if self.literal: print(f'\nK:\n{K}')



        #  ----- ITERATION STATE OUTPUT

        #  get next iteration state
        x_corrected = self.get_corrected_state(x_curr_pred, K, Y, H)
        if self.literal: print(f'\nX_CORRECTED_STATE:\n{x_corrected}')


        #  get next iteration process covariance
        I = np.identity(K.shape[0])
        P_corrected = self.get_corrected_process_covariance(I, K, H, P_pred)
        if self.literal: print(f'\nP_CORRECTED_STATE:\n{P_corrected}')



        #  DEV - delete
        self.kf_results['x_pos']['idx'].append(self.iter)
        self.kf_results['x_pos']['predicted'].append(x_curr_pred[0, 0])
        self.kf_results['x_pos']['measured'].append(self.current_state[0, 0])
        self.kf_results['x_pos']['kalman'].append(x_corrected[0, 0])

        self.kf_results['x_vel']['idx'].append(self.iter)
        self.kf_results['x_vel']['predicted'].append(x_curr_pred[1, 0])
        self.kf_results['x_vel']['measured'].append(self.current_state[1, 0])
        self.kf_results['x_vel']['kalman'].append(x_corrected[1, 0])



        #  set next iteration variables
        self.x_prev = x_corrected
        self.P_prev = P_corrected

