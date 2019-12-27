import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


cp = np.array([ 9.48769027, 22.75068204,  9.48769027, 22.75068204,  9.48769027, 17.10417688])

depth = np.array([0, 1.5, 2, 2.5, 6, 8, 11.0])



x = np.linspace(0,11,20)
y =  np.piecewise(x, [0 < x <= 1.5 ,
                      1.5 < x <= 2,
                      2 < x <= 2.5,
                      2.5 < x <= 6,
                      6 < x <= 8, 8 < x <= 11 ]
                      , cp)
plt.plot(x,y)
plt.show()
