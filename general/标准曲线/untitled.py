import os,sys
sys.path.append(os.path.abspath('..'))

print('importing')
from util import Array,show_regress
import matplotlib.pyplot as plt

c=Array([0,0.2,0.4,0.8,1.2,1.6,2])
A=Array([1e-4,0.0866,0.1735,0.3085,0.4233,0.5379,0.7731])

show_regress(c,A,
            xlabel=r'$\mathit{c} / \mathrm{ mol\cdot L^{-1}}$',ylabel=r'$A$',
            xlim='min',ylim='min',verbose=2,show=False)

plt.savefig('fig.jpg') #保存图片
plt.show()