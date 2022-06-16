import itertools
import numpy as np
import matplotlib.pyplot as plt

step=50 #慎重选择数量
x=np.linspace(-5,5,step)
y=np.linspace(-5,5,step)
grid = list(itertools.product(x,y))

x1,y1=np.meshgrid(x,y)

#球
z = [np.sqrt(max(0, np.power(5,2) -np.square(x) - np.square(y))) for x,y in grid]
z = np.array(z).reshape(50,50)


ax = plt.figure().add_subplot(projection='3d')

surf = ax.plot_surface(x1, y1, z, rstride=1, cstride=1, shade=False)

ax.view_init(10,10)

plt.show()