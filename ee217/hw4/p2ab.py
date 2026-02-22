import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# (a) Hadamard matrix generation (Sylvester construction)
# ============================================================
def hadamard_sylvester(n):
    """
    Generate H_n where n is a power of 2, entries in {+1,-1}.
    Sylvester recursion:
        H_1 = [1]
        H_2n = [[H_n,  H_n],
               [H_n, -H_n]]
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of 2")
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H,  H],
                      [H, -H]])
    return H


def circ_autocorr_pm1(x_pm1):
    """
    Circular autocorrelation r[k] = sum_n x[n] * x[(n+k) mod N]
    """
    x = np.asarray(x_pm1, dtype=np.float64)
    N = len(x)
    r = np.zeros(N, dtype=np.float64)
    for k in range(N):
        r[k] = np.sum(x * np.roll(x, k))
    return r


# ============================================================
# (b) Fast Hadamard Transform (FHT), no library calls
# ============================================================
def fht(x):
    """
    Iterative Fast Hadamard Transform.
    Computes y = H_N x (unnormalized), H_N entries ±1.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    N = x.size
    if N < 1 or (N & (N - 1)) != 0:
        raise ValueError("Length must be a power of 2")

    h = 1
    while h < N:
        step = h * 2
        for i in range(0, N, step):
            a = x[i:i+h].copy()
            b = x[i+h:i+step].copy()
            x[i:i+h] = a + b
            x[i+h:i+step] = a - b
        h = step
    return x


# ============================================================
# Robust loader (tries np.loadtxt first; if file has brackets,
# falls back to stripping them) — still "uses loadtxt" normally.
# ============================================================
def load_vec64(path):
    """
    Load a 64-element vector.

    1) Try np.loadtxt() exactly like HW3.
    2) If it fails due to bracket formatting (e.g. starts with [[),
       fallback to stripping [,], commas and parsing numbers.
    """
    try:
        v = np.loadtxt(path)
        v = np.asarray(v).reshape(-1)
    except Exception:
        with open(path, "r") as f:
            text = f.read()
        text = text.replace("[", " ").replace("]", " ").replace(",", " ")
        v = np.fromstring(text, sep=" ")
        v = np.asarray(v).reshape(-1)

    if v.size != 64:
        raise ValueError(f"{path}: expected 64 samples, got {v.size}")

    return v


# Gaussian model for touch profile across drive lines
def gauss(x, A, mu, sigma, b):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + b


def fit_gaussian(x_mm, y):
    """
    Fit Gaussian using scipy if available, otherwise do a simple grid-search fallback.
    """
    try:
        from scipy.optimize import curve_fit  # type: ignore
        p0 = [
            float(np.max(y) - np.min(y)),
            float(x_mm[np.argmax(y)]),
            10.0,
            float(np.min(y)),
        ]
        params, _ = curve_fit(gauss, x_mm, y, p0=p0, maxfev=20000)
        return params, "scipy.curve_fit"
    except Exception:
        # fallback: coarse grid search over mu, sigma; solve A,b by least squares each time
        mu_grid = np.linspace(x_mm.min(), x_mm.max(), 257)
        sigma_grid = np.linspace(2.0, 40.0, 97)

        best = None
        best_err = np.inf

        for mu in mu_grid:
            for sigma in sigma_grid:
                g = np.exp(-(x_mm - mu)**2 / (2 * sigma**2))
                M = np.column_stack([g, np.ones_like(g)])
                sol, *_ = np.linalg.lstsq(M, y, rcond=None)
                A, b = sol
                yhat = A * g + b
                err = np.mean((y - yhat) ** 2)
                if err < best_err:
                    best_err = err
                    best = (A, mu, sigma, b)

        return best, "grid_search_fallback"


def main():
    # ============================================================
    # Problem 2(a): H64 + circular autocorrelation of 23rd row
    # ============================================================
    N = 64
    H64 = hadamard_sylvester(N)

    row_idx_1based = 23
    row = H64[row_idx_1based - 1, :]
    rxx = circ_autocorr_pm1(row)

    plt.figure()
    plt.stem(np.arange(N), rxx)  # no use_line_collection (new matplotlib)
    plt.title(f"(a) Circular autocorrelation of Hadamard H64 row {row_idx_1based}")
    plt.xlabel("Shift k")
    plt.ylabel("r_xx[k]")
    plt.grid(True)
    plt.show()

    offpeak_max = np.max(np.abs(rxx[1:]))
    print("\n=== Problem 2(a) ===")
    print(f"Row {row_idx_1based}: r[0] = {rxx[0]:.1f}, max |off-peak| = {offpeak_max:.1f}")
    print("Interpretation: Hadamard rows are mutually orthogonal, but a single row’s\n"
          "circular autocorrelation typically has strong structured sidelobes, so it\n"
          "is not very white-noise-like (white noise would look close to a delta).")

    # Optional: check which row has the smallest max off-peak (just for curiosity)
    offpeaks = []
    for i in range(N):
        ri = circ_autocorr_pm1(H64[i, :])
        offpeaks.append(np.max(np.abs(ri[1:])))
    best_row = int(np.argmin(offpeaks)) + 1
    print(f"Row with smallest max off-peak autocorr: row {best_row}, max |off-peak| = {min(offpeaks):.1f}")

    # ============================================================
    # Problem 2(b): FHT decode + touch location
    # ============================================================
    no_touch_path = "./HW4.HCapNoTouch.txt"
    touch_path = "./HW4.HCapTouch.txt"

    y_no = load_vec64(no_touch_path)
    y_t = load_vec64(touch_path)

    # Decode:
    # If y = (1/N) * H c  (or y = H c depending on convention),
    # then applying H again recovers N*c (since H^T H = N I).
    # Our FHT computes H*y, so c_hat = (H*y)/N.
    c_no = fht(y_no) / N
    c_t = fht(y_t) / N

    delta = c_t - c_no

    # Often touch appears as a dip; invert to make it positive
    y_fit = -delta

    # Drive line positions: 2 mm spacing
    x_mm = 2.0 * np.arange(N)

    params, method = fit_gaussian(x_mm, y_fit)
    A, mu, sigma, b = params

    print("\n=== Problem 2(b) ===")
    print(f"Gaussian fit method: {method}")
    print(f"Estimated touch location: mu = {mu:.3f} mm")
    print(f"(sigma = {sigma:.3f} mm, A = {A:.4f}, baseline b = {b:.4f})")

    # Plot decoded touch signature + fit
    plt.figure()
    plt.plot(x_mm, y_fit, marker='o', label="-(c_touch - c_no_touch)")
    x_dense = np.linspace(x_mm.min(), x_mm.max(), 800)
    plt.plot(x_dense, gauss(x_dense, A, mu, sigma, b), label="Gaussian fit")
    plt.title("(b) Touch location from FHT decode + Gaussian fit")
    plt.xlabel("Drive line position (mm)")
    plt.ylabel("Response (arb)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Optional: plot raw decoded vectors
    plt.figure()
    plt.plot(x_mm, c_no, marker='o', label="c_no_touch")
    plt.plot(x_mm, c_t, marker='o', label="c_touch")
    plt.title("(b) Decoded capacitance per drive line (from FHT)")
    plt.xlabel("Drive line position (mm)")
    plt.ylabel("Capacitance (arb)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()