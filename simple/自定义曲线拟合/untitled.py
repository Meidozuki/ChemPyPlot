from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

#Antoine方程  $lg\ p=A-{B \over T+C}$
def func(indep_vars,A,B,C):
    T=indep_vars
    
    return A-B/(T+C)

#以水的饱和蒸气压为例
t=np.array([0,10,20,30,40,50,60,70])
p=np.array([610,1228,2338,4242,7375,12333,19910,31157])

lgp=np.log10(p*(760/101e3))#转化为mmHg并取对数

var,co=optimize.curve_fit(func,xdata=t,ydata=lgp, maxfev=9999)
print(var)
pred=func(t,*var)
real=func(t,8.07,1730,233) #文献值 

plt.plot(t,lgp,label='measure')
plt.plot(t,pred,label='pred')
plt.plot(t,real,label='real')
plt.legend()
plt.show()