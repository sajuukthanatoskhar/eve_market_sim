import math
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

import numpy as np

def f(mu, sigma2, x):
    """
    https://medium.com/analytics-vidhya/kalman-filters-a-step-by-step-implementation-guide-in-python-91e7e123b968
    :param mu:
    :param sigma2:
    :param x:
    :return:
    """
    coefficient = 1.0/math.sqrt(2.0*math.pi*sigma2)
    exponential = math.exp(-0.5 * (x-mu) ** 2 / sigma2)
    return coefficient * exponential

def update(mean1, var1, mean2, var2):
    """
    Update step for 1D kalman filter
    https://medium.com/analytics-vidhya/kalman-filters-a-step-by-step-implementation-guide-in-python-91e7e123b968
    :param mean1:
    :param var1:
    :param mean2:
    :param var2:
    :return:
    """
    new_mean = (var2*mean1 + var1*mean2)/(var2+var1)
    new_var  = 1/(1/var2 + 1/var1)

    return [new_mean, new_var]

def predict(mean1, var1, mean2, var2):
    """
    Predict Step for 1D Kalman Filter
    https://medium.com/analytics-vidhya/kalman-filters-a-step-by-step-implementation-guide-in-python-91e7e123b968
    :param mean1:
    :param var1:
    :param mean2:
    :param var2:
    :return:
    """
    new_mean = mean1 + mean2
    new_var= var1 + var2

    return [new_mean, new_var]


class KalmanFilter:
    def __init__(self, process_noise, measurement_noise, estimate_error):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimate_error = estimate_error
        self.posterior = estimate_error
        self.posterior_estimate = 0

    def update(self, measurement):
        prior = self.posterior_estimate
        self.posterior = 1 / (1 / self.posterior + 1 / self.measurement_noise)
        self.posterior_estimate = prior + self.posterior * (measurement - prior)  # warning: check maths
        self.posterior_estimate = self.posterior - self.posterior * self.posterior / (
                    self.posterior + self.process_noise)  # check maths
        self.posterior = self.posterior - self.posterior * self.posterior / (
                    self.posterior + self.process_noise)  # check maths

    def predict(self):
        return self.posterior_estimate



if __name__ == '__main__':
    # measurements for mu and motions, U
    measurements = [5., 5., 5., 6., 4.]
    motions = [1., 0, 0., 1., -2.]

    # initial parameters
    measurement_sig = 4.
    motion_sig = 2.
    mu = 0.
    sig = 10000.


    for n in range(len(measurements)):
        mu, sig = update(mu, sig, measurements[n], measurement_sig)
        print('Update: [{}, {}]'.format(mu, sig))
        mu, sig = predict(mu, sig, motions[n], motion_sig)
        print('Predict: [{}, {}]'.format(mu, sig))

# print the final, resultant mu, sig
print('\n')
print('Final result: [{}, {}]'.format(mu, sig))

## Print out and display the final, resulting Gaussian
# set the parameters equal to the output of the Kalman filter result
mu = mu
sigma2 = sig

# define a range of x values
x_axis = np.arange(-20, 20, 0.1)

# create a corresponding list of gaussian values
g = []
for x in x_axis:
    g.append(f(mu, sigma2, x))

# plot the result
plt.plot(x_axis, g)
plt.show()
