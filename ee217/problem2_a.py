from problem1_a import bits_to_pm1, lfsr_prbs
from p1_extension import cir_autocorr
from scipy.optimize import curve_fit
import numpy as np 
import matplotlib.pyplot as plt 

# get data 
x = np.loadtxt("./HW3.Pr3.notouch.txt")
x_touch = np.loadtxt("./HW3.Pr3.touch.txt")

prbs511  = lfsr_prbs(n_bits=9,  tap_mask=0x110, seed=257)
c = bits_to_pm1(prbs511)

r = cir_autocorr(x, c)
r_touch = cir_autocorr(x_touch, c)

idx = np.argsort(np.abs(r))[-5:][::-1]
idx_sorted = np.sort(idx)
idx_touch = np.argsort(np.abs(r_touch))[-5:][::-1]
idx_touch_sorted = np.sort(idx_touch)

r_no_peaks = r[idx_sorted]
r_t_peaks  = r_touch[idx_sorted]
delta = np.abs(r_t_peaks) - np.abs(r_no_peaks)
y = -delta
pos = np.array([5, 10, 15, 20, 25])

def gauss(x, A, mu, sigma, b):
    return A*np.exp(-(x-mu)**2/(2*sigma**2)) + b

p0 = [y.max(), 17.5, 5.0, 0.0]   # initial guess
params, _ = curve_fit(gauss, pos, y, p0=p0)
A, mu, sigma, b = params
print("Touch located at: ", f"{mu:.3f}", "mm")

# extension
amps = r[idx_sorted] / len(c)
x_hat = np.zeros_like(x)
for A, k in zip(amps, idx_sorted):
    x_hat += A * np.roll(c, k)
residual = x - x_hat
print("Noise std per sample:", np.std(residual))

# print peaks + pos
print(f"{'Position (mm)':>12} | {'Offset k':>8} | {'No-touch r[k]':>15} | {'Touch r[k]':>15}")
print("-"*65)
for pos_mm, k in zip(pos, idx_sorted):
    print(f"{pos_mm:12.0f} | {k:8d} | {r[k]:15.6f} | {r_touch[k]:15.6f}")


