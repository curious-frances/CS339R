from problem1_a import bits_to_pm1, lfsr_prbs
import numpy as np 
import matplotlib.pyplot as plt 

def cir_autocorr(pm1):
    x = np.array(pm1, dtype=int) 
    N = len(x)
    r = np.zeros(N, dtype=int)
    for k in range(N):
        r[k] = np.sum(x * np.roll(x, -k))
    return r 

prbs511  = lfsr_prbs(n_bits=9,  tap_mask=0x110, seed=0b000000001)
x511 = bits_to_pm1(prbs511)
print("x511 unique:", np.unique(x511))
print("x511 sum:", np.sum(x511))
r511 = cir_autocorr(x511)

print("r[0]", r511[0])
print("unique values in r", np.unique(r511))

plt.figure()
plt.stem(r511)
plt.title('Circular autocorrelation of PRBS511')
plt.xlabel("Shift K")
plt.ylabel("r[k]")
plt.show()
    