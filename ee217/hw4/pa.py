import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1) First-order sigma-delta
# ==============================
def first_order_sigma_delta(x, A=1):
    N = len(x)
    y = np.empty(N, dtype=np.float64)
    v = 0.0
    y_prev = 0.0

    for n in range(N):
        v = v + (x[n] - A*y_prev)
        y_n = 1.0 if v >= 0 else -1.0
        y[n] = y_n
        y_prev = y_n

    return y


# ==============================
# 2) Interpolation filter
# ==============================
def design_interp_fir(num_taps, fs_out, cutoff_hz, osr):
    """
    Windowed-sinc interpolation LPF.
    DC gain scaled to OSR to reconstruct amplitude after zero-stuffing.
    """
    n = np.arange(num_taps, dtype=np.float64)
    m = (num_taps - 1) / 2.0
    fc = cutoff_hz / fs_out

    h = 2.0 * fc * np.sinc(2.0 * fc * (n - m))
    h *= np.hamming(num_taps)

    h /= np.sum(h)
    h *= osr
    return h


def upsample_and_interpolate(x, osr, h):
    xu = np.zeros(len(x) * osr)
    xu[::osr] = x

    gd = (len(h) - 1) // 2
    y_full = np.convolve(xu, h, mode="full")
    y = y_full[gd:gd + len(xu)]
    return y


# ==============================
# 3) SNR calculation
# ==============================
def compute_snr(y, fs, tone_hz):
    N = len(y)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(N, 1/fs)

    P = (np.abs(Y)**2) / (N**2)
    P[1:-1] *= 2  # single-sided

    df = fs / N
    k_tone = int(np.round(tone_hz / df))
    k_max = int(np.floor(48000 / df))

    # Signal power (sum small neighborhood)
    P_signal = np.sum(P[k_tone-2:k_tone+3])

    # Noise power (DC..48k excluding tone region)
    mask = np.ones(k_max+1, dtype=bool)
    mask[k_tone-6:k_tone+7] = False
    P_noise = np.sum(P[:k_max+1][mask])

    snr_db = 10*np.log10(P_signal / P_noise)
    return snr_db, freqs, P


# ==============================
# 4) Main
# ==============================
def main():

    fs_in = 48000.0
    OSR = 128
    fs = fs_in * OSR
    Nfft = 65536
    N_in = Nfft // OSR

    # ---- Coherent 1 kHz tone ----
    df = fs / Nfft
    k = int(np.round(1000 / df))
    tone_hz = k * df

    print(f"FFT bin spacing df = {df:.2f} Hz")
    print(f"Using coherent tone bin k={k}, tone_hz={tone_hz:.2f} Hz")

    # ---- Generate 16-bit two's complement stream ----
    n = np.arange(N_in)
    amp = 10**(-12/20)  # -12 dBFS
    x_float = amp * np.sin(2*np.pi*tone_hz*n/fs_in)

    x_int16 = np.int16(np.round(x_float * 32767))
    x = x_int16.astype(np.float64) / 32768.0

    # ---- Interpolation filter (required per HW) ----
    num_taps = 1023
    cutoff = 20000  # slightly below 24k
    h = design_interp_fir(num_taps, fs, cutoff, OSR)

    x_os = upsample_and_interpolate(x, OSR, h)
    assert len(x_os) == Nfft

    # ---- Sigma-delta ----
    y = first_order_sigma_delta(x_os, A=0.300)

    # ---- SNR ----
    snr_db, freqs, P = compute_snr(y, fs, tone_hz)

    print("\n=== RESULT (Part 1a) ===")
    print(f"SNR (dB) = {snr_db:.2f}")

    # ---- Plot spectrum ----
    P_db = 10*np.log10(P + 1e-20)

    plt.figure()
    idx = freqs <= 250000
    plt.plot(freqs[idx], P_db[idx])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.title(f"Sigma-Delta Output Spectrum (SNR={snr_db:.2f} dB)")
    plt.grid(True)
    plt.show()
    
    # ---- Part (b): Sweep A ----
    A_vals = np.linspace(0.05, 2.0, 40)
    snr_vals = []

    print("\n=== PART (b) : Gain Sweep ===")

    for A in A_vals:
        y = first_order_sigma_delta(x_os, A)
        snr_db, freqs, P = compute_snr(y, fs, tone_hz)
        snr_vals.append(snr_db)
        print(f"A = {A:.3f}   SNR = {snr_db:.2f} dB")

    snr_vals = np.array(snr_vals)

    best_index = np.argmax(snr_vals)
    best_A = A_vals[best_index]
    best_snr = snr_vals[best_index]

    print("\n=== BEST RESULT (Part b) ===")
    print(f"Best A   = {best_A:.3f}")
    print(f"Max SNR  = {best_snr:.2f} dB")


if __name__ == "__main__":
    main()