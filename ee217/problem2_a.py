from problem1_a import bits_to_pm1, lfsr_prbs
from p1_extension import cir_autocorr
import numpy as np 
import matplotlib.pyplot as plt 

# get data 
x = np.loadtxt("./HW3.Pr3.notouch.txt")
x = x - np.mean(x) # remove any dc offset 
x_touch = np.loadtxt("./HW3.Pr3.touch.txt")
x_touch = x_touch - np.mean(x_touch) # remove any dc offset

plt.figure()
plt.plot(x)
plt.title("No-touch waveform (raw)")
plt.xlabel("Sample index n")
plt.ylabel("ADC/sample value")
plt.show()

plt.figure()
plt.plot(x_touch)
plt.title("touch waveform (raw)")
plt.xlabel("Sample index n")
plt.ylabel("ADC/sample value")
plt.show()

prbs511  = lfsr_prbs(n_bits=9,  tap_mask=0x110)
c = bits_to_pm1(prbs511)

r = cir_autocorr(x, c)
r_touch = cir_autocorr(x_touch, c)

idx = np.argsort(np.abs(r))[-5:][::-1]
idx_sorted = np.sort(idx)
idx_touch = np.argsort(np.abs(r_touch))[-5:][::-1]
idx_touch_sorted = np.sort(idx_touch)

print("touch peaks:", list(idx_touch_sorted))
print("touch peak vals:", list(r_touch[idx_touch_sorted]))

# extension
amps = r[idx_sorted] / len(c)
x_hat = np.zeros_like(x)
for A, k in zip(amps, idx_sorted):
    x_hat += A * np.roll(c, -k)
residual = x - x_hat
print("Noise std per sample:", np.std(residual))


