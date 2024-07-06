import os
import numpy as np
from scipy.ndimage import convolve

import matplotlib.pyplot as plt

def g_h(data, x0, dx, g, h, dt):

    """
    Performs g-h filter on 1 state variable with a fixed g and h.

    'data' contains the data to be filtered.
    'x0' is the initial value for our state variable
    'dx' is the initial change rate for our state variable
    'g' is the g-h's g scale factor
    'h' is the g-h's h scale factor
    'dt' is the length of the time step 
    """

    x_est = x0
    results = []
    for z in data:
        # prediction step
        x_pred = x_est + (dx*dt)
        dx = dx

        # update step
        residual = z - x_pred
        dx = dx + h * (residual) / dt
        x_est = x_pred + g * residual
        results.append(x_est)
    return np.array(results)

def discrete_bayes_predict(pdf, offset, kernel):
    prior = convolve(np.roll(pdf, offset), kernel, mode='wrap')
    return prior

def discrete_bayes_normalize(value):
    return value/sum(value)

def discrete_bayes_update(likelihood, prior):
    return discrete_bayes_normalize(likelihood * prior)


if __name__=='__main__':
    pass