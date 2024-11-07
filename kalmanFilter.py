import numpy as np
from numba import jit
from globalVars import global_params

@jit(nopython=True)
def predict_step(x, u, A, B, P, Q):
    x = A @ x + B @ u
    P = A @ P @ A.T + Q
    return x, P

class KalmanFilter:
    def __init__(self):
        dt = global_params['dt']
        u_x = global_params['u_x']
        u_y = global_params['u_y']
        std_acc = global_params['std_acc']
        x_std_meas = global_params['x_std_meas']
        y_std_meas = global_params['y_std_meas']

        self.x = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.u = np.array([[u_x], [u_y]], dtype=np.float64)

        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=np.float64)
        self.B = np.array([[0.5 * (dt ** 2), 0],
                           [0, 0.5 * (dt ** 2)],
                           [dt, 0],
                           [0, dt]], dtype=np.float64)

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)

        self.Q = np.array([[0.25 * (dt ** 4), 0, 0.5 * (dt ** 3), 0],
                           [0, 0.25 * (dt ** 4), 0, 0.5 * (dt ** 3)],
                           [0.5 * (dt ** 3), 0, dt ** 2, 0],
                           [0, 0.5 * (dt ** 3), 0, dt ** 2]], dtype=np.float64) * std_acc ** 2

        self.R = np.array([[x_std_meas ** 2, 0],
                           [0, y_std_meas ** 2]], dtype=np.float64)

        self.P = np.eye(4, dtype=np.float64)

    def predict(self):
        self.x, self.P = predict_step(self.x, self.u, self.A, self.B, self.P, self.Q)

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        return self.x
