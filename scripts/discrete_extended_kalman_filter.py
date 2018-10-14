"""

Extended kalman filter (EKF) localization sample

author: Atsushi Sakai (@Atsushi_twi)

modified by Ben Ware

"""
import numpy as np
import math
import matplotlib.pyplot as plt
import random

# # Estimation parameter of EKF
Q = np.diag([0.1, 0.1, 0.1, 0.1])**2
R = np.diag([1.0, 1.0])**2
DT = 1  # Descrete time steps
SIM_TIME = 100

def get_rand(): return random.choice([1,1,1,0])

show_animation = True

## Assumptions and notes:
# Robot sensor noise is simulated by adding to each coordinate: np.random.randn()//2 
# Input noise is simulated with a 25% chance to not register an input. (unrealistic, but simple)
# Robot does not have prior knowledge of chance of "slipping"
# Robot does not assume it is confined to a descrete domain.
# Simulated input ommits [0,-1] to create simple tendency to move upward

def simulate_input():
    [v_i,v_j] = random.choice([[1,0],[-1,0],[0,1]]) # omitted [0,-1] to create tendency to move upward
    u = np.matrix([v_i, v_j]).T #u is 1x2
    return u

def sumulate_measurement(i,j,noise=True):
    if noise is True:
        # add noise to gps i-j
        i += np.random.randn()//2 
        j += np.random.randn()//2 
    z = np.matrix([i, j])
    return z

def simulate_robot_motion_noise(udi,udj,noise=True):
    if noise is True:
#         effectively: a 25% chance to not register an input
        udi *= get_rand()
        udj *= get_rand()
    ud = np.matrix([udi, udj]).T
    print
    return ud
    
def observation(xTrue, xd, u): # xTrue is 1x4, xd shape is 1x4, u shape is 1x4
    xTrue = motion_model(xTrue, u) 
    
    # add noise to gps i-j
    z = sumulate_measurement(xTrue[0, 0],xTrue[1, 0]) #z is 1x2
    
    # add noise to input
    ud = simulate_robot_motion_noise(u[0, 0],u[1, 0]) #ud is 1x2

    xd = motion_model(xd, ud) #xd is 1x4

    return xTrue, z, xd, ud


def motion_model(x, u):
#     u is the change in location or "velocity"
#     x = position i,j
#     velocities are independent of each other
    F = np.matrix([[1.0, 0, 0, 0],
                   [0, 1.0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])

    B = np.matrix([[1.0, 0],
                   [0, 1.0],
                   [1.0, 0],
                   [0, 1.0]])

    x = F * x + B * u
    return x
def observation_model(x):
    #  Observation Model
    H = np.matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
#         [1, 0],
#         [0, 1]
    ])
    z = H * x
    return z


def jacobF(x, u):
    """
    Jacobian of Motion Model

    motion model
    x_{t+1} = x_t+v_x
    y_{t+1} = y_t+v_y
    x_t{t+1} = v_x
    y_t{t+1} = v_y
    so
    dx/dv_x = 1 
    dy/dv_y = 1
    """
    v_x =u[0, 0]                
    v_y =u[1, 0]                
    jF = np.matrix([ 
         [1.0, 0.0, 1, 0],
         [0.0, 1.0, 0, 1],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]])
    return jF



def jacobH(x):
    # Jacobian of Observation Model
    jH = np.matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    return jH



def ekf_estimation(xEst, PEst, z, u):

    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacobF(xPred, u)
    PPred = jF * PEst * jF.T + Q

    #  Update
    jH = jacobH(xPred)
    zPred = observation_model(xPred)
    y = z.T - zPred
    S = jH * PPred * jH.T + R
    K = PPred * jH.T * np.linalg.inv(S)
    xEst = xPred + K * y
    PEst = (np.eye(len(xEst)) - K * jH) * PPred

    return xEst, PEst


def plot_covariance_ellipse(xEst, PEst):
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
    R = np.matrix([[math.cos(angle), math.sin(angle)],
                   [-math.sin(angle), math.cos(angle)]])
    fx = R * np.matrix([x, y])
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")


def main():
#     print(__file__ + " start!!")
    time = 0.0

    # State Vector [x y yaw v]'
    xEst = np.matrix(np.zeros((4, 1)))
    xTrue = np.matrix(np.zeros((4, 1)))
    PEst = np.eye(4)

    xDR = np.matrix(np.zeros((4, 1)))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((1, 2))

    while SIM_TIME >= time:
        time += DT
        u = simulate_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.vstack((hz, z))

        if show_animation:
            plt.cla()
            plt.plot(hz[:, 0], hz[:, 1], ".g")
            plt.plot(np.array(hxTrue[0, :]).flatten(),
                     np.array(hxTrue[1, :]).flatten(), "-b")
            plt.plot(np.array(hxDR[0, :]).flatten(),
                     np.array(hxDR[1, :]).flatten(), "-k")
            plt.plot(np.array(hxEst[0, :]).flatten(),
                     np.array(hxEst[1, :]).flatten(), "-r")
            plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

if __name__ == '__main__':
    main()
