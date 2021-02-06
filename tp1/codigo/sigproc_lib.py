# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#import cmath
from collections import defaultdict
from math import pi as pi
from  sympy import symbols,Poly

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.pyplot import (axhline, axvline, figure, grid, margins, plot,
                               show, stem, subplot, title, xlabel, xlim,
                               xscale, ylabel, cm)
from numpy import angle, arange, array, asarray, cos, log10, pi, tan, unwrap
from scipy import misc, signal
from scipy.fft import fft, irfft, rfft
from scipy.signal import (bessel, bilinear, buttap, butter, cheb2ap, cheb2ord,
                          cheby1, cheby2, ellip, freqz, lfilter, tf2zpk,
                          zpk2tf, lti)
from  pprint import pprint
#from zplane import zplane


# %%
def gera_seno(A, f, fs, phi, N: int, plot=False) -> (np.ndarray, np.ndarray):

    assert f >= 0, f"Frequencia F={f}Hz NÂO é maior ou igual a zero."
    assert fs > 2*f, f"Frequencia F={f}Hz NÂO é maior que frequencia de Nyquist Fn={2*f}Hz."
    assert N > 0, f"Numero de amostras N={N}Hz NÂO é maior que zero."

    tempo_final = N/fs
    w = 2*pi*f
    t = np.linspace(0, tempo_final, int(N+1),endpoint=True)
    theta = w*t
    x = A*np.sin(theta+phi)
    if f == 0:
        x = np.ones(len(x))*A
    if plot is True:
        plt.stem(t, x)
        plt.ylabel("y(t)")
        plt.xlabel("t(s)")
        plt.grid()
        plt.show()
        print(type(x))
    return (x, t)


def test_gera_seno():
    A = 3
    f = 400
    fs = 1000
    phi = 0*pi/180
    N = 25

    gera_seno(A, f, fs, phi, N)

# %%
def plotsin(data, t=None, ylabel="y(t)", xlabel="t(s)", title="", space="linspace", save=None):
    if t is None and not type(data) is list:
        t = np.linspace(0, len(data), len(data))
    figure, ax = plt.subplots(1)
    if type(data) is list:
        for series in data:
            ax.plot(t, series)
    else:
        ax.plot(t, data)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid()
    plt.show()
    if not save is None:
        figure.savefig(save+title+".png", dpi=figure.dpi)
    return figure, ax


def test_plotsin():
    fs = 500
    (x, t) = gera_seno(A=1, f=0.5, fs=500, phi=0, N=1000)
    plotsin(x, t, ylabel="x[n]", xlabel="nT(s)", save="teste.png")
    
# %%
def plotlog(data, t=None, ylabel="y(t)", xlabel="t(s)", title=None, space="loglog"):

    if t is None and not type(data) is list:
        t = np.logspace(1, len(data), len(data))
    figure, ax = plt.subplots(1)
    log_types = {"semilogy": ax.semilogy,
                 "semilogx": ax.semilogx,
                 "loglog": ax.loglog}
    log_type = log_types[space]
    if isinstance(data,list):
        for series in data:
            log_type(t, series)
    else:
        log_type(t,data)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid()
    if title != None:
        plt.title(title)
    plt.show()
    return figure, ax


def test_plotlog():
    t = np.linspace(100, 1e9)

    plotlog(t, t, ylabel="x[n]", xlabel="nT(s)", space="semilogy")

# %%
def plot_stem(data, t=None, ylabel="y[n]", xlabel="n"):
    if t is None:
        plt.stem(data)
    else:
        plt.stem(t, data)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid()
    plt.show()


def test_plot_stem():
    h = np.array([3, 2.5, 1, 0, -1, 0, 0.5])
    plot_stem(h, ylabel="h[n]", xlabel="n")


def remov_zeros(x):
    return [i for i in x if i != 0]



# %%
def conv(x, h, plot=False):
    length_x = len(x)
    length_h = len(h)
    x = np.append(x, np.zeros(length_h))
    x_rev = np.append(x, np.zeros(length_h))
    h = np.append(h, np.zeros(length_x))
    print("x =", (x))
    print("h =", (h))
    print('\n')
    print('\n')
    N = length_x+length_h
    y = np.zeros(N)
    n = np.linspace(0, N)
    #x_rev = [x[-i] for i in range(1,len(x)+1)]
    for n in range(N):
        for t in range(length_x+1):
            y[n] += x[n-t] * h[t]
            str = ' '*n
    if plot:
        plt.stem(y)
        plt.ylabel("y[n]")
        plt.xlabel("n")
        plt.grid()
        plt.show()
    return y


def test_conv():
    x = np.array([1, 2, 3, 2, 2, 1])
    h = np.array([3, 2.5, 1, 0, -1, 0, 0.5])
    y = conv(x, h)
    print("Funcionou" if (y == [3, 8.5, 15, 15.5, 13, 8, 2, 0, -0.5, 0, 1, 0.5, 0]).all() else "não funcionou")
# %%


def gera_janela(inicio_janela, fim_janela, fs, plot=False):

    fim_janela = 3
    fim_janela_n = fim_janela*fs
    inicio_janela = 1
    inicio_janela_n = inicio_janela*fs
    N = fs * fim_janela
    h = np.zeros(N)
    h[inicio_janela_n:fim_janela_n] = 1
    if plot:
        plt.plot(np.linspace(0, 3, len(h)), h)
        plt.show()

    return h


def test_gera_janela():
    gera_janela(inicio_janela=1, fim_janela=3, fs=500, plot=True)


# %%
def conv(x, h):
    length_x = len(x)
    length_h = len(h)
    x = np.append(x, np.zeros(length_h))
    x_rev = np.append(x, np.zeros(length_h))
    h = np.append(h, np.zeros(length_x))
    #print("x =",x)
    #print("h =",h)
    # print('\n')
    # print('\n')
    N = length_x+length_h
    y = np.zeros(N)
    n = np.linspace(0, N)
    #x_rev = [x[-i] for i in range(1,len(x)+1)]
    for n in range(N):
        for t in range(length_x+1):
            y[n] += x[n-t] * h[t]
            str = ' '*n
        #print("x="+'     '*n, x_rev)
        #print("h="+'     '*(N-1),h)
        #print("y="+'     '*(N-1),y)
        # print('\n')
    # plt.stem(y)
    # plt.show()
    return y


# %%
def prod(x, h):
    length_x = len(x)
    length_h = len(h)
    x = np.append(x, np.zeros(length_h))
    x_rev = np.append(x, np.zeros(length_h))
    h = np.append(h, np.zeros(length_x))
    #print("x =",x)
    #print("h =",h)
    # print('\n')
    # print('\n')
    N = length_x+length_h
    y = np.zeros(N)
    n = np.linspace(0, N)
    #x_rev = [x[-i] for i in range(1,len(x)+1)]
    for n in range(N):
        for t in range(length_x+1):
            y[n] += x[n-t] * h[t]
            str = ' '*n
        #print("x="+'     '*n, x_rev)
        #print("h="+'     '*(N-1),h)
        #print("y="+'     '*(N-1),y)
        # print('\n')
    # plt.stem(y)
    # plt.show()
    return y


# %%
def generate_fourrier_decomposition(a, f=25e3, kn=0, symetric=False):
    A = a
    kn = len(A)
    # phi=np.angle(a)%180
    phi = np.zeros(len(a))
    karray = np.linspace(0, len(A)+1, len(A))

    fs = f * 10000
    if symetric is True:
        Ak = [ak[k] + ak[-k] for k in karray]
    else:
        Ak = A
    N = int(2 * fs//f)  # num amostras
    senos = list()
    for k in range(kn):
        senos.append(gera_seno(A[k], k*f, fs, phi[k]+pi/2, N))

    t = senos[0][1]
    result = np.zeros(N+1)
    for k in range(kn):
        result = result + senos[k][0]
        plt.plot(t, senos[k][0], label=f'k={k}')
    plt.plot(t, result, label=f'resultante')
    print(len(t[0:len(result)]))
    plt.grid()
    plt.ylabel("y")
    plt.xlabel("t (us)")
    plt.show()
# %%


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[0:n-1] = [a[x]/n for n in range(n)]
    ret[n - 1:] = ret[n - 1:] / n
    return ret


# %%
def running_mean(x, N):
    cumsum = numpy.cumsum(numpy.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

#%%
def composicao_sinal(A, phi=None, f=0):
    if isinstance(type(phi), type(None)) and phi is None:
        phi= np.zeros(len(A))
    karray=np.linspace(0,len(A)+1,len(A))
    num_harmonicos = len(A)
    fs= 10*len(A)*f  # 10x maior que o harmônico final
    N= int(fs//f) # num amostras
    print(f"Número de amostras necessário é N={N}")
    seno_series_y_x = [gera_seno(A[k], k*f, fs, (phi[k]+pi/2), N) for k in range(num_harmonicos)]
    series_y = [data[0] for data in seno_series_y_x]
    series_x = [data[1] for data in seno_series_y_x]
    result = np.sum(series_y, axis=0) 
    series_y.append(result)
    plotsin(data=series_y, t=series_x[0]) 
    return result, series_x[0] 
    
def test_composicao_sinal():
    A=[2.8, 6, 2.6]
    phi=[0, - pi/4,  +3*pi/8]
    f= 25e3
    _,_ = composicao_sinal(A=A, phi=phi, f=f)


#%%



def caracterizacao_de_LTI(transfer_function, frequency_range):
    w = np.linspace(frequency_range[0],frequency_range[1],10000)

    mag, phase=transfer_function(w)
    plotlog(data=mag, t=w, ylabel="Magnitude (dB)", xlabel="Frequência(rad/s)",space="semilogx", title="Resposta de magnitude em função da frequencia")

    plotlog(data=phase*180/pi, t=w, ylabel="Fase(º)", xlabel="Frequência(rad/s)",space="semilogx", title="Resposta de fase em função da frequencia")



    tau_c = -1*np.diff(phase)/np.diff(w)
    #tau_c = np.where(tau_c>0,tau_c, 0)
    x = w[0:len(tau_c)//6]
    y=tau_c[0:len(tau_c)//6]
    plotsin(data=y, t=x, ylabel="Atraso de grupo(s)", xlabel="Frequência(rad/s)", title="Atraso de grupo")

def transf_1(w):
    y = (1e6j * w + 5e6)/ (-np.square(w) + 6e3j*w +2.5e7)
    mag = 20*np.log10(np.abs(y))
    phase = np.angle(y) 
    return mag, phase
    
    
def test_caracterizacao_de_LTI():
    caracterizacao_de_LTI(transf_1,[1e2,3e5])
    
    
# %%  
def plot_zpk(zeros, poles, k):
    t1 = plt.plot(zeros.real, zeros.imag, 'o', markersize=10.0, alpha=0.9)
    t2 = plt.plot(poles.real, poles.imag, 'x', markersize=10.0, alpha=0.9)
    grid(True, color = '0.7', linestyle='-', which='major', axis='both')
    grid(True, color = '0.9', linestyle='-', which='minor', axis='both')
    title('Poles and zeros')
    mark_overlapping(zeros)
    mark_overlapping(poles)
    plt.ylabel("Im")
    plt.xlabel("Re")

def mark_overlapping(items):
    """
    Given `items` as a list of complex coordinates, make a tally of identical 
    values, and, if there is more than one, plot a superscript on the graph.
    """
    d = defaultdict(int)
    for i in items:
        d[i] += 1
    for item, count in d.items():
        if count > 1:
            plt.text(item.real, item.imag, r' ${}^{' + str(count) + '}$', fontsize=13)

def test_plot_zph():
    #teste
    num = np.poly1d([1,2])
    d1= np.poly1d([1,1])
    d2= np.poly1d([1,3])
    den = d1*d2
    sys = signal.TransferFunction(num,den)
    z,p,k = signal.tf2zpk(num,den)
    plot_zpk(z,p,k)
# %%   
    
def bode_from_tf(num,dem,start,end):
    start_freq, end_freq, samples=(start,end,1000)
    s1 = signal.lti(num,dem)
    w, mag, phase = s1.bode(w=np.logspace(start_freq,end_freq,samples))
    plt.subplot(2, 1, 1)
    plt.semilogx(w, mag)    # Bode magnitude plot
    plt.ylabel("Magnitude |H| dB")
    plt.grid(True, color = '0.7', linestyle='-', which='major', axis='both')
    plt.subplot(2, 1, 2)
    plt.semilogx(w, phase)  # Bode phase plot
    plt.grid(True, color = '0.7', linestyle='-', which='major', axis='both')
    plt.ylabel("angulo(H) (º)")
    plt.xlabel("Frequência w (rad/s)")
    plt.show()
    
def test_bode_from_tf():
    coef_denominador = [1, 2, 4]
    coef_numerador= [1, 0]
    bode_from_tf(coef_numerador, coef_denominador, -1,2)
    




#test_plot_phase_surf()
# %%  3d phase plot

def abs_surf_plotter(abs_surf,X,Y,yrang,xrang,zrang):
    import mpmath
    import pylab
    from mpl_toolkits.mplot3d import Axes3D
    mpmath.dps = 5
    #%matplotlib widget
    fig = pylab.figure()
    ax = Axes3D(fig)
    
    #ax.set_zlim3d(0, 100)
    #ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.jet)
    Z = np.clip(abs_surf,zrang[0],zrang[1])
    #Z = abs_surf
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
    #ax.contourf(X, Y, W, zdir='z', offset=-1, cmap=plt.cm.hot)
    ax.set_zlim(zrang[0], zrang[1])
    #ax.plot_wireframe(X, Y, W, rstride=5, cstride=5)
    ax.set_ylabel("Im{s}")
    ax.set_xlabel("Re{s}")
    ax.set_zlabel("|H(s)|")
    ax.set_title("S plane amplitude response")     
    pylab.show()
 
def phase_surf_plotter(phase_surf,X,Y,yrang,xrang,zrang,angle="degree"):
    import mpmath
    import pylab
    from mpl_toolkits.mplot3d import Axes3D
    mpmath.dps = 5
    #%matplotlib widget
    
    #%matplotlib widget
    limits = [-180,180] if angle=="degree" else [-pi,pi]
    fig = pylab.figure()
    ax = Axes3D(fig)
    #Z = np.clip(phase_surf,limits[0],limits[1])
    Z = phase_surf
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
    #ax.contourf(X, Y, W, zdir='z', offset=-1, cmap=plt.cm.hot)
    #ax.set_zlim(limits[0],limits[1])
    #ax.plot_wireframe(X, Y, W, rstride=5, cstride=5)
    ax.set_ylabel("Im{s}")
    ax.set_xlabel("Re{s}")
    ax.set_zlabel("angle(H(s))")
    ax.set_title("S plane phase response")     
    pylab.show()
 
def plot_surf(num,den,yrang,xrang,zrang,samples=50, angle="degree"):

    X = np.arange(xrang[0], xrang[1], (xrang[1]-xrang[0])/samples)
    Y = np.arange(yrang[0], yrang[1], (yrang[1]-yrang[0])/samples)
    X, Y = np.meshgrid(X, Y)
    xn, yn = X.shape
    abs_surf = X*0
    phase_surf = X*0
    
    rad_to_degree = 180/pi if angle=="degree" else 1
    for xk in range(xn):
        for yk in range(yn):
            z = complex(X[xk,yk],Y[xk,yk])
            H_z = np.polyval(num,z)/np.polyval(den,z)
            abs_surf[xk,yk] = np.abs(H_z)
            phase_surf[xk,yk] = np.angle(H_z)*rad_to_degree
            
    import mpmath
    import pylab
    from mpl_toolkits.mplot3d import Axes3D
    mpmath.dps = 5
    #%matplotlib widget
    abs_surf_plotter(abs_surf,X,Y,yrang,xrang,zrang)
    
    #%matplotlib widget
    phase_surf_plotter(phase_surf,X,Y,yrang,xrang,zrang,angle="degree")
    
def test_plot_surf():
    z = []
    k= 2.4873413456829807e+23
    p = np.array([ -489.3491+2143.9785j , -1371.1257+1719.3373j ,
       -1981.334  +954.16016j, -2199.1147   -0.j     ,
       -1981.334  -954.16016j, -1371.1257-1719.3373j ,
        -489.3491-2143.9785j ], dtype=np.complex64)
    num, dem = zpk2tf(z,p,k )    
    plot_surf(num,dem,xrang=[-3000,-500],yrang=[-3000,3000],zrang=[0,100],samples=100,angle="rad")
   
#test_plot_surf()






# %%    interative 3d magnitude plot
   
def plot_abs_surf(num,den,yrang,xrang,zrang,samples=50):

    X = np.arange(xrang[0], xrang[1], (xrang[1]-xrang[0])/samples)
    Y = np.arange(yrang[0], yrang[1], (yrang[1]-yrang[0])/samples)
    X, Y = np.meshgrid(X, Y)
    xn, yn = X.shape
    abs_surf = X*0
    phase_surf = X*0

    for xk in range(xn):
        for yk in range(yn):
            z = complex(X[xk,yk],Y[xk,yk])
            H_z = np.polyval(num,z)/np.polyval(den,z)
            abs_surf[xk,yk] = np.abs(H_z)
            
    abs_surf_plotter(abs_surf,X,Y,yrang,xrang,zrang)    
    
def test_plot_abs_surf():
    coef_denominador = [1, 2, 4]
    coef_numerador= [1, 0]
    plot_abs_surf(coef_numerador,coef_denominador,xrang=[-2,0],yrang=[-5,5],zrang=[0,10])
    
#test_plot_abs_surf()

# %%    interative 3d phase plot
   
def plot_phase_surf(num,den,yrang,xrang,zrang,samples=50,angle="degree"):

    X = np.arange(xrang[0], xrang[1], (xrang[1]-xrang[0])/samples)
    Y = np.arange(yrang[0], yrang[1], (yrang[1]-yrang[0])/samples)
    X, Y = np.meshgrid(X, Y)
    xn, yn = X.shape
    abs_surf = X*0
    phase_surf = X*0
    rad_to_degree = 180/pi if angle=="degree" else 1

    for xk in range(xn):
        for yk in range(yn):
            z = complex(X[xk,yk],Y[xk,yk])
            H_z = np.polyval(num,z)/np.polyval(den,z)
            phase_surf[xk,yk] = np.angle(H_z)*rad_to_degree
            
    phase_surf_plotter(phase_surf,X,Y,yrang,xrang,zrang,angle=angle)
    
def test_plot_phase_surf():
    coef_denominador = [1, 2, 4]
    coef_numerador= [1, 0]
    plot_phase_surf(coef_numerador,coef_denominador,xrang=[-2,0],yrang=[-5,5],zrang=[0,10])
 
#test_plot_phase_surf()

# %%    sym tf
def sym_tf(num,den):
    s = symbols('s')
    tf = Poly(num,s)/ Poly(den,s)
    tf
    return tf
  
def test_sym_tf():
    sym_tf(
        [0.001, 0, 0.008, 0, 0.008],
        [1,0.7746,0.3,0.0684,0.008 ]
    )
test_sym_tf()


# %%
def stable_poles(poles):
    return poles[np.where(poles<= 0)]

def test_stable_poles():
    poles = np.array([-0.707317+0.01724349j, -0.707317-0.01724349j,
            0.707317+0.01724349j,  0.707317-0.01724349j])
    zeros = np.array([-0.707317+0.01724349j, -0.707317-0.01724349j,
            0.707317+0.01724349j])
    stable_poles(poles)
    numerator, denominator = zpk2tf(zeros,poles,1)
# %%

def tf(num,dem):
        return lambda s:(np.polyval(num,s)/np.polyval(dem,s))
def test_tf():
    poles = np.array([-0.707317+0.01724349j, -0.707317-0.01724349j,
            0.707317+0.01724349j,  0.707317-0.01724349j])
    zeros = np.array([-0.707317+0.01724349j, -0.707317-0.01724349j,
            0.707317+0.01724349j])
    numerator, denominator = zpk2tf(zeros,poles,1)

    tf(numerator,denominator)(0)


#%%
if __name__ == "__main__":
    test_gera_janela()
    test_gera_seno()
    test_plot_stem()
    test_conv()
    test_plotsin()
    test_plotlog()
    test_composicao_sinal()
    test_caracterizacao_de_LTI()
    test_plot_zph()
    test_plot_surf()
    test_plot_abs_surf()
    test_plot_phase_surf()
    test_tf()
    test_stable_poles()

# %%
