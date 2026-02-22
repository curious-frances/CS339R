import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1) 1.5-bit quantizer (3-level) + first-order loop
# ==============================
def quantize_1p5(v, thr=0.33):
    """
    1.5-bit quantizer:
      if v > +thr -> +1
      if v < -thr -> -1
      else -> 0
    Thresholds are ±0.33 full-scale per HW. :contentReference[oaicite:1]{index=1}
    """
    if v > thr:
        return 1.0
    elif v < -thr:
        return -1.0
    else:
        return 0.0

def first_order_sigma_delta_1p5(x, A=1.0, thr=0.33):
    """
    First-order loop with 3-level DAC feedback:
        v[n] = v[n-1] + x[n] - A*y[n-1]
        y[n] = Q_1p5(v[n]) in {-1, 0, +1}
    """
    N = len(x)
    y = np.empty(N, dtype=np.float64)
    v = 0.0
    y_prev = 0.0

    for n in range(N):
        v = v + x[n] - A * y_prev
        y_n = quantize_1p5(v, thr=thr)
        y[n] = y_n
        y_prev = y_n

    return y


# ==============================
# 2) Interpolation filter (same as your earlier)
# ==============================
def design_interp_fir(num_taps, fs_out, cutoff_hz, osr):
    n = np.arange(num_taps, dtype=np.float64)
    m = (num_taps - 1) / 2.0
    fc = cutoff_hz / fs_out
    h = 2.0 * fc * np.sinc(2.0 * fc * (n - m))
    h *= np.hamming(num_taps)
    h /= np.sum(h)
    h *= osr
    return h

def upsample_and_interpolate(x, osr, h):
    xu = np.zeros(len(x) * osr, dtype=np.float64)
    xu[::osr] = x
    gd = (len(h) - 1) // 2
    y_full = np.convolve(xu, h, mode="full")
    y = y_full[gd:gd + len(xu)]
    return y


# ==============================
# 3) SNR calculation (same metric as HW)
# ==============================
def compute_snr(y, fs, tone_hz):
    N = len(y)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(N, 1/fs)

    P = (np.abs(Y)**2) / (N**2)
    if len(P) > 2:
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

    snr_db = 10*np.log10(P_signal / (P_noise + 1e-300))
    return snr_db, freqs, P


# ==============================
# 4) Main (Part c: repeat a+b)
# ==============================
def main():
    fs_in = 48_000.0
    OSR = 128
    fs = fs_in * OSR
    Nfft = 65_536
    N_in = Nfft // OSR

    # Coherent tone bin near 1kHz (avoid leakage)
    df = fs / Nfft
    k = int(np.round(1000.0 / df))
    tone_hz = k * df

    print(f"FFT bin spacing df = {df:.2f} Hz")
    print(f"Using coherent tone bin k={k}, tone_hz={tone_hz:.2f} Hz")

    # 16-bit two's complement sine at -12 dBFS
    n = np.arange(N_in)
    amp = 10**(-12/20)
    x_float = amp * np.sin(2*np.pi*tone_hz*n/fs_in)
    x_int16 = np.int16(np.round(x_float * 32767))
    x = x_int16.astype(np.float64) / 32768.0

    # Interpolation filter ahead of modulator (required)
    num_taps = 1023
    cutoff = 20000
    h = design_interp_fir(num_taps, fs, cutoff, OSR)
    x_os = upsample_and_interpolate(x, OSR, h)
    assert len(x_os) == Nfft

    # ---- Part (c-a/b): sweep A to maximize SNR ----
    thr = 0.33  # per HW :contentReference[oaicite:2]{index=2}
    A_vals = np.linspace(0.05, 2.0, 40)

    print("\n=== PART (c): 1.5-bit quantizer sweep ===")
    best_A = None
    best_snr = -np.inf
    best_freqs = None
    best_P = None

    snr_vals = []
    for A in A_vals:
        y = first_order_sigma_delta_1p5(x_os, A=A, thr=thr)
        snr_db, freqs, P = compute_snr(y, fs, tone_hz)
        snr_vals.append(snr_db)
        print(f"A = {A:.3f}   SNR = {snr_db:.2f} dB")

        if snr_db > best_snr:
            best_snr = snr_db
            best_A = A
            best_freqs = freqs
            best_P = P

    print("\n=== BEST RESULT (Part c) ===")
    print(f"Best A   = {best_A:.3f}")
    print(f"Max SNR  = {best_snr:.2f} dB")

    # Plot SNR vs A (useful for writeup)
    plt.figure()
    plt.plot(A_vals, snr_vals, marker='o')
    plt.xlabel("Feedback gain A")
    plt.ylabel("SNR (dB)")
    plt.title("Part (c): SNR vs A (1.5-bit quantizer)")
    plt.grid(True)
    plt.show()

    # Plot FFT spectrum at best A
    P_db = 10*np.log10(best_P + 1e-20)
    plt.figure()
    idx = best_freqs <= 250_000
    plt.plot(best_freqs[idx], P_db[idx])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.title(f"Part (c): Output Spectrum (best A={best_A:.3f}, SNR={best_snr:.2f} dB)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()