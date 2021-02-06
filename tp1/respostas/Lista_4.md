# Victor Moraes - 2016027600
![q1](imgs/q1.png)  



$$ |H(j\omega)|^2 = H(j\omega) . H^* (j\omega)$$
$$ |H(j\omega)|^2 = H(j\omega) . H (-j\omega)$$
$$ js=s$$
$$ |H(s)|^2 = H(s) . H (-s)$$
$$ |H(s)|^2 = {1 \over 1 + (s/\omega_c)^N} . {1 \over 1 - (s/\omega_c)^N}$$
### a) Função de transferência:
$$ H(s) = {1 \over 1 + (s/\omega_c)^N}$$
$$ H(s) = {1 \over 1 + (s/700.\pi)^7}$$
a) Polos:
$$ 1+ (s/\omega_c)^N = 0$$  
$$ (s/\omega_c)^N = (-1)$$
$$ s = \omega_c (-1)^{1/N}  $$
$$ s = 700\pi(-1)^{1/7}  $$
$$ p_k = -700\pi \angle {2\pi k \over 7},  k \in [0,6]$$

Zeros: não possui zeros do polinômio do numerador.


b)

```python
import numpy as np
from math import pi as pi
import cmath
from scipy import signal, misc
from scipy.fft import fft, rfft, irfft
import matplotlib.pyplot as plt
from zplane import zplane
from matplotlib import patches
from matplotlib.pyplot import axvline, axhline
from collections import defaultdict
from scipy.signal import (freqz, butter, bessel, cheby1, cheby2, ellip,
                            tf2zpk, zpk2tf, lfilter, buttap, bilinear, cheb2ord, cheb2ap
                            )
from numpy import asarray, tan, array, pi, arange, cos, log10, unwrap, angle
from matplotlib.pyplot import (stem, title, grid, show, plot, xlabel,
                                ylabel, subplot, xscale, figure, xlim,
                                margins)
```

```python

num = np.poly1d([1])
wc = (700*pi)
den= np.poly1d([1,0,0,0,0,0,0,1])

sys = signal.TransferFunction(num,den)
z,p,k = signal.tf2zpk(num,den)
p=np.singlecomplex(wc*p)
[print(f"Polo {i} = {round(p[i])}") for i, pole in enumerate(p)]
plot_zpk(z,p,k)
```
```python
Polo 0 = (-2199+0j)
Polo 1 = (-1371+1719j)
Polo 2 = (-1371-1719j)
Polo 3 = (489+2144j)
Polo 4 = (489-2144j)
Polo 5 = (1981+954j)
Polo 6 = (1981-954j)
```
![1_b_2](imgs/1_b_2.png)

```python
tf_mag = lambda x: float(20*log(1/(1 + (x/(700*pi))**14), 10))
f = np.logspace(2,5,1000)

mag = np.vectorize(tf_mag)(f)

plt.xlim([1e2,1e4])
plt.xlabel("Frequência w (rad/s)")
plt.ylim([-100,10])
plt.ylabel("Magnitude (dB)")
plt.grid(True, color = '0.7', linestyle='-', which='major', axis='both')
plt.semilogx(f, mag)    # Bode magnitude plot
```

$$ \prod x_i$$

![b_mag](imgs/b_mag.png)

![q2](imgs/q2.png)



![q3](imgs/q3.png)