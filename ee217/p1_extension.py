from problem1_a import bits_to_pm1, lfsr_prbs
import numpy as np 
import matplotlib.pyplot as plt 

def cir_autocorr(x_pm1, y_pm1):
    x = np.array(x_pm1, dtype=int)
    y = np.array(y_pm1, dtype=int)
    N = len(x)
    r = np.zeros(N, dtype=int)
    for k in range(N):
        r[k] = np.sum(x * np.roll(y, -k))
    return r

seqA  = lfsr_prbs(n_bits=9,  tap_mask=0x110)
seqB  = lfsr_prbs(n_bits=9,  tap_mask=0x108)

print({seqA == seqB})

XA = bits_to_pm1(seqA)
XB = bits_to_pm1(seqB)

rAB = cir_autocorr(XA, XB)

print("max cross correlation: ", np.max(rAB))
print("min cross correlation: ", np.min(rAB))


plt.figure()
plt.stem(rAB)
plt.title("Circular cross-correlation: seqA (0x110) vs seqB (0x108)")
plt.xlabel("Shift k")
plt.ylabel("r_AB[k]")
plt.show()