import numpy as np
import matplotlib.pyplot as plt

lam=np.array([440,460,480,490,500,506,508,510,520,530,540,560])
A=np.array([4904,5680,6535,6643,6866,7041,7085,7105,6717,5438,3694,1307])*1e-4

plt.plot(lam,A)
plt.xlabel('$\lambda$ /nm')
plt.ylabel('A')
plt.show()