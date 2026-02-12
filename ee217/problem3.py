import numpy as np 
import matplotlib.pyplot  as plt 

# Consts
a = 0.8

def DFFT(w, a):
    return (1 - a**2) / (1 - 2*a*np.cos(w) + a**2)

# Plot X(w) for 0 <= w <= 2pi
w = np.linspace(0, 2*np.pi, 2000, endpoint=True)
Xw = DFFT(w, a)

plt.figure()
plt.plot(w, Xw)
plt.title("X(ω) for x[n] = a^{|n|}, a=0.8")
plt.xlabel("ω (rad/sample)")
plt.ylabel("X(ω)")
plt.show()

def ifft_samples(N, a):
    k = np.arange(N)
    w_k = 2*np.pi*k/N
    Xk = DFFT(w_k, a)
    xN = np.fft.ifft(Xk).real  
    return xN

for N in [20, 100]:
    xN = ifft_samples(N, a)

    plt.figure()
    plt.stem(np.arange(N), xN)
    plt.title(f"IFFT of sampled X(ω), N={N}")
    plt.xlabel("n")
    plt.ylabel("x_N[n]")
    plt.show()