import math
import numpy as np
from numpy import linalg
from scipy import stats
from scipy import constants as cons
from scipy.stats import linregress
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
# from PIL import Image

def Array(x,dtype=None):
    return np.array(x,dtype=dtype)

def Like(x):
    return range(len(x))

def vectorize(*args,f):
    return np.vectorize(f)(*args)

def new_vector(arg,shape=None,like=None,pos=None):
    #用nan填充到指定长度vector
    if shape is None: shape=like.shape[0]
    assert shape is not None
    
    if pos is None: pos=shape//2
    re=[np.nan]*pos
    re.append(arg)
    re+=[np.nan]*(shape-pos-1)
    return np.expand_dims(re,axis=0)

def slice_vector(v,slices):
    re=[]
    for i in slices:
        re.append(v[i])
    return np.array(re)

def print_lists(*args,var=locals().items()):
    for lis in args:
        for k,v in var:
            if v is lis:
                print(k,end=' = ')
        print(lis)
        
def form_print(x,format_s=None):
    if format_s is None:
        print(*x,sep='\t')
    else:
        for i in x:
            print(format_s % i,end='')
        print()

def quick_axis_label(xlabel=None,ylabel=None):
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
        
def self_adaptive_axis_range(x=None,y=None,bias_coef=0.01):
    def cal_range(x):
        mi,ma=min(x),max(x)
        R=ma-mi
        bias=R*bias_coef
        return mi-bias,ma+bias
    
    if x is not None:
        plt.xlim(*cal_range(x))
    if y is not None:
        plt.ylim(*cal_range(y))

def show_data_frame(data,index=None,columns=None,formatter=None,subset=None,index_name=None,columns_name=None):
    if not isinstance(data,np.ndarray): data=np.array(data)
    assert len(data.shape) == 2
    if index is None:
        index=range(1,data.shape[0]+1)
    df=pd.DataFrame(data,index=index,columns=columns)
    df.index.name=index_name
    df.columns.name=columns_name
    if formatter: df=df.style.format(formatter,subset=subset)
    return df

def arrange_formatter(*args):
    global columns
    i=1
    re={}
    while (i < len(args)):
        k,v=args[i-1],args[i]
        if type(k) is int:
            re[columns[k]]=v
        elif isinstance(k,(list,range)):
            for index in k:
                re[columns[index]]=v
        i+=2
    print(re)
    return re

def nan_white(df):
    def nan_white_fn(x):
        if np.isnan(x):
            return 'color:white'
        else:
            return 'color:black'
        
    try:
        return df.style.applymap(nan_white_fn)
    except:
        return df.applymap(nan_white_fn)
    
def slice_column(df,idx):
    idx=[df.columns[i] for i in idx]
    return df[idx]

def br(s,step=15):
    ls=list(s)
    i=1
    while i*step-1 < len(s):
        ls.insert(i*step-1,'<br>')
        i+=1
    return ''.join(ls)

def at(df,i):
    return np.array(df[df.columns[i]])

def show_regress(x,y,degree=1,force=False,
                 scatter=True,clip_on=False,
                 extend_to_axis=False,extend=None,
                 xlabel=None,ylabel=None,plot_label=None,
                 xlim=None,ylim=None,
                 verbose=1,show=True):
    '''
    拟合数据x,y并且绘制所拟合曲线
    degree为多项式次数，默认为1
    可选参数：
    xlabel,ylabel为坐标轴标签，plot_label为曲线标签
    scatter为True（默认）时绘制原始x,y散点,clip_on传入scatter
    xlim和ylim支持快捷格式'min','max','minmax'，以及如同plt.xlim((left,right))原始参数
    extend_to_axis为True则将曲线绘制到x与y轴的截距，extend为自由增加x取值
    '''
    
    def set_axis_range(feature,limit=None,**kwargs):
        import logging
        def check_format_str(s):
            words=['min','max','minmax']
            if isinstance(s,str) and s in words:
                return True
            else:
                logging.warning(f"str only support for {words},but received {s}")
                return False
            
        if feature not in ['x','y']:
            raise ValueError
        fn_dict={'x':(plt.xlim,x), 'y':(plt.ylim,y)}
        fn,var=fn_dict[feature]
        if isinstance(limit,tuple):
            fn(limit)
        elif isinstance(limit,str):
            if check_format_str(limit) is False:
                return
            elif limit == 'min':
                fn((min(var),None))
            elif limit == 'max':
                fn((None,max(var)))
            elif limit == 'minmax':
                fn((min(var),max(var)))
    
    global reg
    #线性回归且非强制多项式
    if degree == 1 and not force:
        #线性回归输出参数以及回归方程和拟合优度
        reg=linregress(x,y)
        if verbose > 0:
            print(reg)
            if verbose > 1:
                print('y={:.3e}x {:+.3e}'.format(reg.slope,reg.intercept),end=' \t')
            print("R^2=%.5f" % np.square(reg.rvalue))
    else:
        #多项式回归输出多项式系数
        reg=np.polyfit(x,y,degree)
        print(reg)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if scatter: plt.scatter(x,y,clip_on=clip_on)

    if extend_to_axis:
        x=np.append(x,[0])
        plt.xlim(left=0)
    if extend:
        x=np.append(x,[extend])
        
    #产生回归y值
    if degree == 1 and not force:
        predict=[i*reg.slope+reg.intercept for i in x]
    else:
        predict=[np.polyval(reg,i) for i in x]
    #绘制
    if xlim is not None:
        set_axis_range('x',xlim)
    if ylim is not None:
        set_axis_range('y',ylim)
                
    plt.plot(x,predict,label=plot_label,linewidth=4)
    if show:
        plt.show()
    
def from_standard_curve(y,print_fn=False):
    global reg
    if print_fn:
        for i in y:
            print((i-reg.intercept)/reg.slope)
    else:
        return (y-reg.intercept)/reg.slope

def adj_f(t0,asst,h0):
    h=(t0-h0)
    dt=(t0-asst)
    return t0+1.56e-4*h*dt

def new_c(volume,c0,total_V):
    return c0*volume/total_V

def time_change(expr):
    m,sec=expr//100,expr%100
    assert sec <= 60
    return m*60+sec

def relative_error(pi,p0):
    return abs(pi-p0)/p0*100

def Q_check(data,idx,alpha=0.1):
    Q_table={0.1:[0,0,0,0.94,0.76,0.64,0.56,0.51,0.47,0.44,0.41],
             0.05:[0,0,0,0.97,0.84,0.73,0.64,0.59,0.54,0.51,0.49]}
    upper,lower=np.max(data),np.min(data)
    n=len(data)
    if data[idx] == upper:
        near=np.max(np.delete(data,idx))
    else:
        near=np.min(np.delete(data,idx))
    Q=abs((data[idx]-near)/(upper-lower))
    Q_lim=Q_table[alpha][n]
    print(f'n={n}, Q={Q}, limits={Q_lim}')
    return Q > Q_lim

def cal_Cronbache_alpha(k,var):
    return (k/(k-1))*(var[-1]-sum(var[:-1]))/var[-1]

def t_interval(x,alpha=0.8):
    n=len(x)-1
    avg=np.mean(x)
    std=np.std(x,ddof=1)
    interval=np.array(stats.t.interval(alpha,n))
    re=avg+std/np.sqrt(n)*interval
    print('{:.4f}+/-{:.4f}'.format(avg,re[1]-avg))
    return re