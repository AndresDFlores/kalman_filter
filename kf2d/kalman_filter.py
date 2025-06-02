import numpy as np

class KalmanFilter:

    def __init__(self):
        pass


    def predict_next_state(self, A, x, B, u):

        state_matrix = A@x
        control_matrix = B@u

        return state_matrix+control_matrix
    

    def predict_process_covariance(self, A, P, Q=0):

        '''
            diagonal of output:  variance
            upper right/lower left triangles of output:  covariance
            standard deviation is the squareroot of the variance

            VARIANCE:  all values along a dimension will fall between (avg-variance) and (avg+variance)
            STANDARD DEVIATION:  68% of values fall within (avg-stdv) and (avg+stdv), or within 1 standard deviation
        '''
                
        P_pred = (A@P@np.transpose(A))+Q

        return P_pred
    

    def get_measured_input(self, C, x, z):

        Y = (C@x)+z

        return Y
    

    def get_kalman_gain(self, P, H, R, related_dims=False):
        
        K = (P@H)/((H@P@np.transpose(H))+R)
        
        if not related_dims:
            K = np.diag(np.diag(K))

        return K
    

    def get_corrected_state(self, x, K, Y, H):

        x_corrected = x+K@(Y-(H@x))

        return x_corrected
    

    def get_corrected_process_covariance(self, I, K, H, P):

        P_corrected = (I-(K@H))@P

        return P_corrected