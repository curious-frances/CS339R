import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Part (b) modulator (1-bit, output as ±1) with feedback gain A
# v[n] = v[n-1] + x[n] - A*y[n-1]
# y[n] = sign(v[n]) in {+1,-1}
# ============================================================
def first_order_sigma_delta_1bit(x, A=1.0):
    N = len(x)
    y = np.empty(N, dtype=np.float64)
    v = 0.0
    y_prev = 0.0
    for n in range(N):
        v = v + (x[n] - A * y_prev)
        y_n = 1.0 if v >= 0.0 else -1.0
        y[n] = y_n
        y_prev = y_n
    return y


# ============================================================
# Interpolation filter (48 kHz -> 6.144 MHz by OSR=128)
# zero-stuff + windowed-sinc LPF
# IMPORTANT: scale LPF DC gain by OSR to restore amplitude
# ============================================================
def design_interp_fir(num_taps, fs_out, cutoff_hz, osr):
    n = np.arange(num_taps, dtype=np.float64)
    m = (num_taps - 1) / 2.0
    fc = cutoff_hz / fs_out

    h = 2.0 * fc * np.sinc(2.0 * fc * (n - m))
    h *= np.hamming(num_taps)

    h /= np.sum(h)   # DC gain 1
    h *= osr         # DC gain OSR after zero-stuff
    return h


def upsample_and_interpolate(x, osr, h):
    xu = np.zeros(len(x) * osr, dtype=np.float64)
    xu[::osr] = x
    gd = (len(h) - 1) // 2
    y_full = np.convolve(xu, h, mode="full")
    y = y_full[gd:gd + len(xu)]  # align + keep same length as xu
    return y


# ============================================================
# SNR metric (for parts a/b): tone peak / integrated noise DC..48k
# ============================================================
def compute_snr(y, fs, tone_hz):
    N = len(y)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(N, 1 / fs)

    P = (np.abs(Y) ** 2) / (N ** 2)
    if len(P) > 2:
        P[1:-1] *= 2.0  # single-sided

    df = fs / N
    k_tone = int(np.round(tone_hz / df))
    k_max = int(np.floor(48_000.0 / df))

    # signal power: small neighborhood
    P_signal = np.sum(P[max(0, k_tone - 2):min(len(P), k_tone + 3)])

    # noise power: DC..48k excluding tone neighborhood
    mask = np.ones(k_max + 1, dtype=bool)
    lo = max(0, k_tone - 6)
    hi = min(k_max, k_tone + 6)
    mask[lo:hi + 1] = False
    P_noise = np.sum(P[:k_max + 1][mask])

    snr_db = 10.0 * np.log10(P_signal / (P_noise + 1e-300))
    return snr_db, freqs, P


# ============================================================
# Part (d): reconstruction LPF at 6.144 MHz + decimate by 128
# ============================================================
def design_decim_fir(num_taps, fs_high, cutoff_hz):
    n = np.arange(num_taps, dtype=np.float64)
    m = (num_taps - 1) / 2.0
    fc = cutoff_hz / fs_high

    h = 2.0 * fc * np.sinc(2.0 * fc * (n - m))
    h *= np.hamming(num_taps)

    h /= np.sum(h)  # unity DC gain
    return h


def lowpass_and_decimate(y_high, h, osr):
    """
    Filter then downsample by osr.
    Returns:
      y_48k: decimated output
      gd_hi: group delay in high-rate samples
    """
    gd_hi = (len(h) - 1) // 2
    yf_full = np.convolve(y_high, h, mode="full")
    yf = yf_full[gd_hi:gd_hi + len(y_high)]  # align to remove group delay
    y_48k = yf[::osr]
    return y_48k, gd_hi


def best_fit_gain(x_ref, x_rec):
    """
    Least-squares gain that maps x_rec -> x_ref (amplitude match).
    """
    N = min(len(x_ref), len(x_rec))
    xr = x_ref[:N]
    yr = x_rec[:N]
    denom = np.dot(yr, yr)
    if denom < 1e-300:
        return 1.0
    return np.dot(xr, yr) / denom


def first_faithful_time_rms(x_ref, x_rec, fs_in=48_000.0, rmse_thresh=0.02, window_ms=2.0):
    """
    Faithful = RMSE over a window is below rmse_thresh.
    More realistic than a per-sample absolute threshold for sigma-delta.

    Returns (idx, t_sec, W, rmse_at_idx) or (None, None, W, None)
    """
    N = min(len(x_ref), len(x_rec))
    e = x_rec[:N] - x_ref[:N]

    W = int(round(window_ms * 1e-3 * fs_in))
    W = max(W, 1)

    for i in range(0, N - W):
        rmse = np.sqrt(np.mean(e[i:i + W] ** 2))
        if rmse < rmse_thresh:
            return i, i / fs_in, W, rmse

    return None, None, W, None


# ============================================================
# Main (runs Part a/b + Part d reconstruction)
# ============================================================
def main():
    # ----- given -----
    fs_in = 48_000.0
    OSR = 128
    fs = fs_in * OSR              # 6.144 MHz
    Nfft = 65_536                 # required FFT size
    N_in = Nfft // OSR            # 512 samples @48k

    # Coherent ~1 kHz tone to avoid leakage in 65536-pt FFT
    df = fs / Nfft
    k = int(np.round(1000.0 / df))
    tone_hz = k * df

    print(f"FFT bin spacing df = {df:.2f} Hz")
    print(f"Using coherent tone bin k={k}, tone_hz={tone_hz:.2f} Hz")

    # ----- 48kHz 16-bit two's complement input sine at -12 dBFS -----
    n = np.arange(N_in)
    amp = 10 ** (-12 / 20)  # -12 dBFS
    x_float = amp * np.sin(2 * np.pi * tone_hz * n / fs_in)

    x_int16 = np.int16(np.round(x_float * 32767.0))
    x_48k = x_int16.astype(np.float64) / 32768.0  # normalized

    # ----- interpolation ahead of modulator (required) -----
    interp_taps = 1023
    interp_cutoff = 20_000.0
    h_i = design_interp_fir(interp_taps, fs_out=fs, cutoff_hz=interp_cutoff, osr=OSR)
    x_os = upsample_and_interpolate(x_48k, OSR, h_i)
    assert len(x_os) == Nfft

    # ============================================================
    # Part (b): sweep A and find best SNR (also answers part a max SNR)
    # ============================================================
    A_vals = np.linspace(0.05, 2.0, 40)
    snrs = []

    best_A = None
    best_snr = -np.inf
    best_freqs = None
    best_P = None

    print("\n=== PART (b): Gain Sweep (1-bit) ===")
    for A in A_vals:
        y = first_order_sigma_delta_1bit(x_os, A=A)
        snr_db, freqs, P = compute_snr(y, fs, tone_hz)
        snrs.append(snr_db)
        print(f"A = {A:.3f}   SNR = {snr_db:.2f} dB")
        if snr_db > best_snr:
            best_snr = snr_db
            best_A = A
            best_freqs = freqs
            best_P = P

    print("\n=== BEST RESULT (Parts a/b) ===")
    print(f"Best A   = {best_A:.3f}")
    print(f"Max SNR  = {best_snr:.2f} dB")

    # Plot spectrum at best A (raw modulator output)
    P_db = 10 * np.log10(best_P + 1e-20)
    plt.figure()
    idx = best_freqs <= 250_000
    plt.plot(best_freqs[idx], P_db[idx])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.title(f"1-bit Output Spectrum (best A={best_A:.3f}, SNR={best_snr:.2f} dB)")
    plt.grid(True)
    plt.show()

    # Plot SNR vs A (for writeup)
    plt.figure()
    plt.plot(A_vals, snrs, marker="o")
    plt.xlabel("Feedback gain A")
    plt.ylabel("SNR (dB)")
    plt.title("SNR vs A (1-bit)")
    plt.grid(True)
    plt.show()

    # ============================================================
    # Part (d): lowpass + decimate the 1-bit output at best A
    # ============================================================
    y_high = first_order_sigma_delta_1bit(x_os, A=best_A)  # ±1 @ 6.144 MHz

    # Reconstruction/decimation LPF:
    # cutoff must be < 24kHz (new Nyquist after decimation to 48k).
    decim_taps = 1023
    decim_cutoff = 20_000.0
    h_d = design_decim_fir(decim_taps, fs_high=fs, cutoff_hz=decim_cutoff)

    x_rec_48k, gd_hi = lowpass_and_decimate(y_high, h_d, OSR)

    # Scale reconstructed to match original amplitude (LS fit)
    g = best_fit_gain(x_48k, x_rec_48k)
    x_rec_scaled = g * x_rec_48k
    print(f"\nBest-fit gain to match amplitude: g = {g:.4f}")

    # Theoretical delay dominated by FIR group delay
    t_gd = gd_hi / fs
    print("\n=== PART (d): Reconstruction ===")
    print(f"Decim LPF taps L={decim_taps}, group delay={gd_hi} high-rate samples")
    print(f"Group delay time ≈ {t_gd * 1e6:.2f} us")
    print(f"Group delay ≈ {t_gd * fs_in:.2f} samples at 48 kHz")

    # Measure “time until faithful” using RMSE criterion
    idx_ok, t_ok, W, rmse = first_faithful_time_rms(
        x_ref=x_48k,
        x_rec=x_rec_scaled,
        fs_in=fs_in,
        rmse_thresh=0.02,
        window_ms=2.0
    )

    if idx_ok is not None:
        print(f"Faithful (RMSE<0.02 over {W} samples) at sample {idx_ok} (@48k) => {t_ok*1e3:.3f} ms, RMSE={rmse:.4f}")
    else:
        print("Did not meet faithful RMSE threshold in this record (try rmse_thresh=0.03–0.05 or shorter window).")

    # Plot original vs reconstructed (scaled) for first few ms
    plt.figure()
    nplot = min(400, len(x_48k), len(x_rec_scaled))
    t = np.arange(nplot) / fs_in
    plt.plot(t, x_48k[:nplot], label="original 48k (input)")
    plt.plot(t, x_rec_scaled[:nplot], label="reconstructed 48k (scaled)", alpha=0.85)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Part (d): Original vs Reconstructed (scaled)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()