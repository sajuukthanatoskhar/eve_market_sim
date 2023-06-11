import scipy
import numpy as np
import control

A = np.array([[1,2],[3,4]])
B = np.array(np.identity(2))
C = np.array(np.identity(2))
D = np.zeros((2,2))

sys_1 = control.StateSpace(A,B,C,D)
A = np.array([[5,6],[7,8]])
sys_2 = control.StateSpace(A,B,C,D)