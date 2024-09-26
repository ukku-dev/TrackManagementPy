import numpy as np
from globalVars import global_params

__all__ = ['KalmanFilter']

class KalmanFilter:
    def __init__(self):
        dt = global_params['dt']
        u_x = global_params['u_x']
        u_y = global_params['u_y']
        std_acc = global_params['std_acc']
        x_std_meas = global_params['x_std_meas']
        y_std_meas = global_params['y_std_meas']

        self.x = np.matrix([[0], [0], [0], [0]])
        self.u = np.matrix([[u_x], [u_y]])

        self.A = np.matrix([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1,  0],
                            [0, 0, 0,  1]])
        self.B = np.matrix([[0.5 * (dt ** 2), 0],
                            [0, 0.5 * (dt ** 2)],
                            [dt, 0],
                            [0, dt]])

        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        self.Q = np.matrix([[0.25 * (dt ** 4), 0, 0.5 * (dt ** 3), 0],
                            [0, 0.25 * (dt ** 4), 0, 0.5 * (dt ** 3)],
                            [0.5 * (dt ** 3), 0, dt ** 2, 0],
                            [0, 0.5 * (dt ** 3), 0, dt ** 2]]) * std_acc ** 2

        self.R = np.matrix([[x_std_meas ** 2, 0],
                            [0, y_std_meas ** 2]])

        self.P = np.eye(4)

    def predict(self):
        self.x = self.A * self.x + self.B * self.u
        self.P = self.A * self.P * self.A.T + self.Q

    def update(self, z):
        y = z - self.H * self.x
        S = self.H * self.P * self.H.T + self.R
        K = self.P * self.H.T * np.linalg.inv(S)
        self.x = self.x + K * y
        I = np.eye(4)
        self.P = (I - K * self.H) * self.P

    def get_state(self):
        return self.x

