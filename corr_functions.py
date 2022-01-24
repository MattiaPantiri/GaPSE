from mcfit import P2xi, xi2P
from scipy.interpolate import interp1d
import numpy as np
from scipy.special import spherical_jn
from math import pi, sin
from scipy.integrate import quad
import hankl

#k, Pk = np.loadtxt('power_spectrum.txt', unpack=True, skiprows=3)
k, Pk = np.loadtxt("file_pk.txt", unpack=True)

r00_array, I00_array = P2xi(k, lowring=True)(Pk, extrap=True)
r20_hankl, I20_hankl = hankl.P2xi(k, Pk, l=0, n=2)
r02_array, I02_array = P2xi(k, l=2, lowring=True)(Pk, extrap=True)
r04_array, I04_array = P2xi(k, lowring=True, l=4)(Pk, extrap=True)
r20_array, I20_array = P2xi(k, lowring=True, n=-2)(Pk, extrap=True)
r40_array, I40_array = P2xi(k, lowring=True, n=-4)(Pk, extrap=True)
r40_hankl, I40_hankl = hankl.P2xi(k, Pk, l=0, n=4)
r31_array, I31_array = P2xi(k, lowring=True, n=-3, l=1)(Pk, extrap=True)
r13_array, I13_array = P2xi(k, lowring=True, n=-1, l=3)(Pk, extrap=True)
#r11_hankl, I11_hankl = hankl.P2xi(k, Pk, l=1, n=1)
r22_array, I22_array = P2xi(k, lowring=True, n=-2, l=2)(Pk, extrap=True)
r22_hankl, I22_hankl = hankl.P2xi(k, Pk, n=2, l=2)
#, ext=1)
r11_array, I11_array = P2xi(k, lowring=True, l=1, n=-1)(Pk, extrap=True)
I02_array = -I02_array
I20_array = (I20_array+100.90505044179127)/(r20_array**2)
I40_array = I40_array/(r40_array**4)
I31_array = -I31_array/(r31_array**3)
I22_array = I22_array/(r22_array**2)
I11_array = I11_array/(r11_array)
I13_array = I13_array/(r13_array)

#r04_array_new = []
#for r in r04_array:
#    if r < 10**5:
#        r04_array_new.append(r)
#    else:
#        break
#l = len(r04_array_new)
#I04_array_new = I04_array[:l]

I00 = interp1d(r00_array, I00_array, fill_value="extrapolate")
I02 = interp1d(r02_array, I02_array, fill_value="extrapolate")
I04 = interp1d(r04_array, I04_array, fill_value="extrapolate")
I20 = interp1d(r20_array, I20_array, fill_value="extrapolate")
I40 = interp1d(r40_array, I40_array, fill_value="extrapolate")
I22 = interp1d(r22_array, I22_array, fill_value="extrapolate")
I11 = interp1d(r11_array, I11_array, fill_value="extrapolate")
I31 = interp1d(r31_array, I31_array, fill_value="extrapolate")
I13 = interp1d(r13_array, I13_array, fill_value="extrapolate")

#r20_array_new = []
#I20_array_new = []
#for i in range(len(r20_array)):
#    r = r20_array[i]
#    if (r < 10**5):
#        r20_array_new.append(r)
#        I20_array_new.append(I20_array[i])

Pk_inter = interp1d(k, Pk)

def integral_new(s):
    result = quad(lambda q: q**2/(2*pi**2) * Pk_inter(q)*(q*s - sin(q*s))/((q*s)**5), k[0], k[-1], epsabs=0.1, epsrel=0.1)
    return result[0]

def integral_log(s):
    result = quad(lambda t: (Pk_inter(np.exp(t))*(-sin(np.exp(t)*s) + np.exp(t)*s))/(2*pi**2*np.exp(2*t)*s**5), np.log(k[0]), np.log(k[-1]))
    return result[0]

#s_bf = np.linspace(1, 1000, num=1000)
#xi_bf = []
#xi_bf_log = []
#[xi_bf_log.append(integral_log(r)) for r in s_bf]
#[xi_bf.append(integral_log(r)) for r in s_bf]

#I40_tilde = interp1d(s_bf, xi_bf, fill_value="extrapolate")

#print(integral_new(100))

#def integral22(s):
#    result = quad(lambda q: q**2/(2*pi**2)*Pk_inter(q)*spherical_jn(2, (q*s))/((q*s)**2), k[0], k[-1], epsabs=0.1, epsrel=0.1)
#    return result[0]

#def integral20(s):
#    result = quad(lambda q: q**2/(2*pi**2)*Pk_inter(q)*sin(q*s)/((q*s)**3), k[0], k[-1], epsabs=0.1, epsrel=0.1)
#    return result[0]

#s_bf = np.logspace(-4, 4, num=1000)
#xi22_bf = []
#[xi22_bf.append(integral22(r)) for r in s_bf]

#xi20_bf = []
#[xi20_bf.append(integral20(r)) for r in s_bf]


#k1, Pk1 = xi2P(r20_array, lowring=True)(I20_array, extrap=True)
#Pk1 = Pk1*k1**2
