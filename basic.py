import numpy as np, \
    scipy, \
    matplotlib.pyplot as plt

class ExtractedCommodity:
    def __init__(self, q0 = 10, p0 = 5, delay = 1):
        self.quantity_total = [q0]
        self.price_total = [p0]
        self.p0 = self.price_total[0]
        self.q0 = self.quantity_total[0]
        self.price_change_delay = delay
    def step(self, q_dot):
        self.quantity_total.append(q_dot + self.quantity_total[-1])
        if self.price_change_delay < len(self.price_total):
            self.price_total.append(q_dot*self.price_total[-1*self.price_change_delay]/self.quantity_total[-1*self.price_change_delay])
        else:
            self.price_total.append(q_dot*self.price_total[0*self.price_change_delay]/self.quantity_total[0*self.price_change_delay])


def sumblk(inputs : list):
    """
    Makes a sum block
    """
    return sum(inputs)

def productblk(inputs : list):
    """
    Make a product block
    """
    return np.prod(inputs)

class differentiateblk:
    """
    A simple differentiating block, default timebase is 1 sec
    """
    def __init__(self):
        self.lastval = None
        self.timebase = 1
        self.outputval = None

    def updateval(self, val):
        """
        Updates the value of the block
        """
        self.outputval = (self.lastval-val)/self.timebase
        return self.outputval

class integrateblk:
    """
    Simple integration block, default timebase is 1 sec
    """
    def __init__(self):
        self.timebase = 1
        self.outputval = None
    def set_timebase(self, newtimebase):
        """
        Sets the time base
        """
        self.timebase = newtimebase

    def updateval(self, val):
        """
        Updates the value and outputs it
        """
        self.outputval = val * self.timebase
        return self.outputval

    def get_output_value(self):
        """
        Gets the output value without doing the update value
        """
        return self.outputval

def generate_white_gaussian_noise(min_value: float, max_value: float, size: int) -> np.ndarray:
    """
    Generates white Gaussian noise.

    Args:
    min_value: The minimum value of the output noise.
    max_value: The maximum value of the output noise.
    size: The number of noise samples to generate.

    Returns:
    An array of shape (size,) containing Gaussian noise with values between min_value and max_value.
    """
    # Generate noise with zero mean and unit variance
    noise = np.random.normal(loc=0.0, scale=1.0, size=size)

    # Scale the noise to the desired range
    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))  # Scale to [0, 1]
    noise = noise * (max_value - min_value) + min_value  # Scale to [min_value, max_value]



    return noise



import numpy as np

def generate_sinusoid_wave(num_samples: int, amplitude: float, frequency: float, phase: float, offset: float) -> np.ndarray:
    """
    Generates a simulated sinusoid wave.

    Args:
    num_samples: The number of samples in the waveform.
    amplitude: The amplitude of the waveform.
    frequency: The frequency of the waveform in Hz.
    phase: The phase of the waveform in radians.
    offset: The DC offset of the waveform.

    Returns:
    An array containing the sampled waveform.
    """
    # Generate time vector
    t = np.arange(num_samples)

    # Generate waveform
    x = amplitude * np.sin(2 * np.pi * frequency * t / num_samples + phase) + offset

    return x


def get_greedGain(p_0, current_p):
    """
    An amazing concept in EVE Online, where when the price of something drops
    below its original price at the start of the sim, people will start to not mine it

    """
    greedGain = (current_p/p_0)**2

    return greedGain


if __name__ == '__main__':
    dt = 1 # for numerical solving :)
    Ttime = list(range(0, 50000, dt))
    q_out = []
    q_out.append(500)
    p_out = []
    p_out.append(100)
    wn_gend = generate_white_gaussian_noise(10, -4, len(Ttime)-1)
    sinusoid = np.zeros(len(Ttime)) #len(Ttime)-1, 10, 0.5, 0, -2)

    price_sys = scipy.signal.StateSpace([0], [-1], [1.1*1], [0])

    greedGain = []
    for t in range(0,len(Ttime)-1):
        greedGain.append(get_greedGain(p_out[0], p_out[-1]))
        q_dot =  (wn_gend[t]+sinusoid[t])*greedGain[-1]
        q_out.append(q_out[-1]+q_dot)

        p_out.append(p_out[-1] - p_out[-1]*q_dot/q_out[-1])







    plt.plot(Ttime, p_out)
    plt.title("Price of Goods")
    plt.figure(2)
    plt.plot(Ttime, q_out)
    plt.title("Quantity of Goods")
    plt.show()




    # qty_sys_x0 = 50
    #
    # tout, yout, xout  = scipy.signal.lsim(qty_sys, wn_gend+sinusoid, Ttime, x0)
    #
    #
    # plt.plot(Ttime, wn_gend+sinusoid, 'r', alpha=0.5, linewidth=1, label='input')
    #
    # plt.plot(tout, yout, 'k', linewidth=1.5, label='output')
    #
    # plt.legend(loc='best', shadow=True, framealpha=1)
    #
    # plt.grid(alpha=0.3)
    #
    # plt.xlabel('t')
    #
    # plt.show()
    #
