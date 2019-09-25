import numpy as p







# ============= PML parameters ============ #
car_dim = 0.1
m = 2
Lpml = 1.0 
R = 10e-8

beta_0 = (m + 1) * V_r * p.log( 1 / abs(R)) / (2 * Lpml)
alpha_0 = (m + 1) * car_dim*  p.log(1 / abs(R)) / (2 * Lpml)
