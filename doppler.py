from scipy.integrate import dblquad, quad
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from math import pi, sqrt, cos, sin, acos, tan, atan
import numpy as np
import time
import random as rnd
import matplotlib.pyplot as plt
from columnar import columnar
import cosmo_functions as cosmo
import corr_functions as corr
import argparse
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description= "Convolution of doppler autocorrelation with an azymuthally symmetric window function")
parser.add_argument('filename', help="Output file name", type = str)
#parser.add_argument('n_mu', help="Number of points for mu integral", type = int)
#parser.add_argument('n_s1', help="Number of points for s1 integral", type = int)
parser.add_argument('n_integrals', help="Number of s values in which the s1, mu integral is computed", type = int)

args = parser.parse_args()

filename = args.filename
#n_mu = args.n_mu
#n_s1 = args.n_s1
n_integrals = args.n_integrals

def doppler_autocorr_white(s, s1, mu):
    "Doppler autocorrelation in the basis s, s1, mu from Martin and Emanuele article"
    z = cosmo.z(s1)
    f = cosmo.f(z)
    D = cosmo.D(z)
    x = s/s1
    a = f**2*D**2*(1+mu*x)*corr.I20(s)*s**2/(3*sqrt(1 + x**2 + 2*mu*x))
    b = f**2*D**2*(1-2*mu*x -3*mu**2)*corr.I22(s)*s**2/(3*sqrt(1 + x**2 + 2*mu*x))
    #H = cosmo.H(z)/(1+z)
    #R = 1 - 1./(H*s1)
    return a+b
    #R**2*

def doppler_autocorr(s1, s2, costheta):
    "Doppler autocorrelation in the basis s1, s2, costheta from Enea and Emanuele article"

    z1 = cosmo.z(s1)
    z2 = cosmo.z(s2)

    s = (s1*s1 + s2*s2 - 2*s1*s2*costheta)**(0.5)
    c1 = 3*s1*s2 - 2*costheta*(s1*s1 + s2*s2) + s1*s2*costheta*costheta
    c2 = (1./3.)*costheta*s*s

    D1 = cosmo.D(z1)
    D2 = cosmo.D(z2)
    f1 = cosmo.f(z1)
    f2 = cosmo.f(z2)
    H1 = cosmo.H(z1)/(1+z1)
    H2 = cosmo.H(z2)/(1+z2)
    R1 = 1 - 1./(H1*s1)
    R2 = 1 - 1./(H2*s2)
    prefac = D1*D2*f1*f2*R1*R2*H1*H2

    return prefac*(c1*((1./45.)*corr.I00(s) + (2./63.)*corr.I02(s) + (1./105.)*corr.I04(s)) + c2*corr.I20(s))

def eff_z(s_min, s_max):
    "Returns the effective redshift for the survey considered"
    V = 0.5*(4./3.)*pi*(s_max**3 - s_min**3)
    return 2*pi/V*quad(lambda s: s**2*cosmo.z(s), s_min, s_max)[0]

def mu_int_eff(s):
    return quad(lambda mu: integrand(s, s1_eff, mu), -1, 1)[0]

def mu_int_eff_F(s):
    "mu integral using quad library, with effective redshift approximation"
    return quad(lambda mu: integrand_F(s, s1_eff, mu), -1, 1)[0]

def mu_int_eff_white(s1eff, s):
    return quad(lambda mu: doppler_autocorr_white(s, s1eff, mu), -1, 1)[0]

def mu_int_eff_white_F(s1eff, s):
    return quad(lambda mu: integrand_F_white(s, s1eff, mu), -1, 1)[0]

def integrand_F_white(s, s1, mu):
    s2 = (s1*s1 + s*s + 2*s*s1*mu)**(0.5)
    h1 = np.heaviside(s2 - r_min, 0.)
    h2 = np.heaviside(r_max - s2, 0.)
    return doppler_autocorr_white(s, s1, mu)*h1*h2*F_inter(mu, s/s1)[0][0]

def integrand(s, s1, mu):
    s2 = sqrt(s1*s1 + s*s + 2*s*s1*mu)
    costheta = (mu*s + s1)/s2
    return doppler_autocorr(s1, s2, costheta)

def integrand_F(s, s1, mu):
    "Change of basis from s1, s2, costheta to s, s1, mu including also the window function"
    s2 = (s1*s1 + s*s + 2*s*s1*mu)**(0.5)
    costheta = (mu*s + s1)/s2
    h1 = np.heaviside(s2 - r_min, 0.)
    h2 = np.heaviside(r_max - s2, 0.)
    return h1*h2*s1*s1*doppler_autocorr(s1, s2, costheta)*F_inter(mu, s/s1)[0][0]

#def mu_s1_integral(s):
#    result = dblquad(lambda mu, s1: integrand(s, s1, mu), r_min, r_max, -1, 1, epsabs=0.01, epsrel=0.01)
#    return result[0]

#def mu_s1_integral_F(s):
#    result = dblquad(lambda mu, s1: integrand_F(s, s1, mu), r_min, r_max, -1, 1, epsabs=0.5, epsrel=0.5)
#    return result[0]

def mu_integral_trap_eff(s):
    "mu integral with trapezoid method, and effective redshift approximation"
    n_points = n_mu
    points = np.linspace(-1, 1, num=n_points)
    coefficients = np.full(n_points, 2)
    coefficients[0] = 1
    coefficients[-1] = 1
    progsum = []
    [progsum.append(coefficients[i]*integrand_F(s, s1_eff, points[i])) for i in range(n_points)]
    total = sum(progsum)
    return total/(n_points-1)

#def mu_integral_trap(s, s1):
#    n_points = n_mu
#    points = np.linspace(-1, 1, num=n_points)
#    coefficients = np.full(n_points, 2)
#    coefficients[0] = 1
#    coefficients[-1] = 1
#    progsum = []
#    [progsum.append(coefficients[i]*integrand_F(s, s1, points[i])) for i in range(n_points)]
#    total = sum(progsum)
#    return total/(n_points-1)

#def s1_integral_trap(s):
#    n_points = n_s1
#    points = np.linspace(r_min, r_max, num=n_points)
#    coefficients = np.full(n_points, 2)
#    coefficients[0] = 1
#    coefficients[-1] = 1
#    progsum = []
#    [progsum.append(coefficients[i]*mu_integral_trap(s, points[i]))  for i in range(n_points)]
#    total = sum(progsum)
#    return 0.5*total*(r_max-r_min)/(n_points-1)

#def mu_s1_integral_MC(s):
#    n_throws = 10000
#    n_blocks = 100
#    block_length = int(n_throws/n_blocks)
#    prog_blkaverage = 0.
#    prog_squared_blkaverage = 0.
#    prog_sum = 0.

#    for i in range(n_blocks):
#        prog_sum = 0.
#        for j in range(block_length):
#            mu = rnd.uniform(-1, 1)
#            s1 = rnd.uniform(r_min, r_max)
#            prog_sum += 2*(r_max-r_min)*integrand_F(s, s1, mu)
#        n = i+1
#        blkaverage = prog_sum/block_length
#        squared_blkaverage = blkaverage**2
#        prog_blkaverage += blkaverage
#        prog_squared_blkaverage += squared_blkaverage
#        average = 1./n * prog_blkaverage
#        squared_average = 1./n * prog_squared_blkaverage
#        if n == 1:
#            error = 0
#        else:
#            error = sqrt(1./(n-1) * (squared_average - average**2))

#    return average




### ----- MAIN ----- ###

# read and interpolate the F(x, mu) funcion
F = np.loadtxt("../F_x_mu/F_x_mu_new.txt", skiprows=3, usecols=2)
x = np.linspace(0, 8, num=80, endpoint=False)
mu = np.linspace(-1, 1, num=20)
F1 = []
for i in range(20):
    appo = F[i*80:(i+1)*80]
    F1.append(appo)
F_inter = RectBivariateSpline(mu, x, F1) # NB the order of the variables: mu, s

#survey specifications
z_min = 0.05
z_max = 0.2

r_min = cosmo.com_dist(z_min)
r_max = cosmo.com_dist(z_max)

#print(r_min)
#print(r_max)

effective = eff_z(r_min, r_max)
print("effective redshift:", effective)

s1_eff = cosmo.com_dist(effective)
print("comoving distance at eff red:", s1_eff)


# compute the integral over mu of the doppler autocorrelation function
s_array = np.logspace(-1, 3, num=n_integrals)
result = []
t0 = time.time()
[result.append(mu_int_eff_F(s)) for s in s_array]
t1 = time.time()
print("elapsed:", t1 - t0)

# write the result in a file
out = open(filename, "w")
data = []
[data.append([s_array[i], result[i]]) for i in range(len(s_array))]
headers = ["s", "xi(s)"]
table = columnar(data, headers = headers, no_borders = True)
out.write(table)
