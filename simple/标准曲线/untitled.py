import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

c=np.array([0,0.2,0.4,0.8,1.2,1.6,2])
A=np.array([1e-4,0.0866,0.1735,0.3085,0.4233,0.5379,0.7731])
x,y=c,A

reg=linregress(x,y)
print('y={:.3e}x {:+.3e}'.format(reg.slope,reg.intercept),end=' \t')
print("R^2=%.5f" % np.square(reg.rvalue))

#绘制散点
plt.scatter(x,y,clip_on=0)
#绘制回归直线
predict=[i*reg.slope+reg.intercept for i in x]
plt.plot(x,predict,linewidth=3)

#设定坐标范围
plt.xlim(left=min(x))
plt.ylim(bottom=min(y))

#设置坐标轴
plt.xlabel(r'$\mathit{c} / \mathrm{ mol\cdot L^{-1}}$')
plt.ylabel(r'$A$')
plt.savefig('fig.jpg') #保存图片
plt.show()