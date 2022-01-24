from scipy.interpolate import interp1d
import input_Castorina
import numpy as np

z_array, H_array, comdist_array, angdist_array, lumdist_array, D_array, f_array = np.loadtxt('cosmology.txt', unpack=True, skiprows=3)
h = input_Castorina.parameters.get('h')
H_array = H_array/h
comdist_array = comdist_array*h
angdist_array = angdist_array*h
lumdist_array = lumdist_array*h

z = interp1d(comdist_array, z_array)
H = interp1d(z_array, H_array)
com_dist = interp1d(z_array, comdist_array)
ang_dist = interp1d(z_array, angdist_array)
lum_dist = interp1d(z_array, lumdist_array)
D = interp1d(z_array, D_array)
f = interp1d(z_array, f_array)
f_r = interp1d(comdist_array, f_array)

H0 = 100/(3*10**5)
OmegaM0 = input_Castorina.parameters.get('Omega_b') + input_Castorina.parameters.get('Omega_cdm')
