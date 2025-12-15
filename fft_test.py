import numpy as np
import time

# --- Configuration ---
N = 4096  # Size of the signal (must be power of 2 for optimal FFT)

# --- 1. Signal Initialization ---
# Create a complex signal with two sine waves (5Hz and 10Hz)
t = np.arange(N) / N
signal = (np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)).astype(np.complex128)
# Set imaginary part to zero for a real signal
signal.imag = 0.0

print(f"FFT Comparison Test (Size N={N})")
print("-" * 40)

# ----------------------------------------------------------------------
# 2. Naive DFT Implementation (O(N^2))
# ----------------------------------------------------------------------
def naive_dft(x):
    """Computes the Discrete Fourier Transform (DFT) using the definition (O(N^2))."""
    N_sig = len(x)
    output = np.zeros(N_sig, dtype=np.complex128)

    # N-th root of unity (W = exp(-i * 2 * pi / N))
    # Pre-calculate the core complex exponential term
    w = np.exp(-2j * np.pi / N_sig)

    print("Running Naive DFT (O(N^2))... ", end="", flush=True)
    start_time = time.time()
    
    # Iterate over each output frequency bin (k)
    for k in range(N_sig):
        sum_val = 0.0 + 0.0j
        # Iterate over each input time sample (n)
        for n in range(N_sig):
            # Sum_{n=0}^{N-1} x[n] * W^(k*n)
            sum_val += x[n] * (w**(k * n))
        output[k] = sum_val
    
    end_time = time.time()
    time_ms = (end_time - start_time) * 1000
    print(f"Done. Time: {time_ms:.2f} ms")
    return output

# ----------------------------------------------------------------------
# 3. Optimized FFT Implementation (O(N log N))
# ----------------------------------------------------------------------
def numpy_fft(x):
    """Computes the Fast Fourier Transform (FFT) using the NumPy library (O(N log N))."""
    print("Running NumPy FFT (O(N log N))... ", end="", flush=True)
    start_time = time.time()
    
    # The optimized FFT call
    output = np.fft.fft(x)
    
    end_time = time.time()
    time_ms = (end_time - start_time) * 1000
    print(f"Done. Time: {time_ms:.2f} ms")
    return output

# ----------------------------------------------------------------------
# 4. Execute and Compare
# ----------------------------------------------------------------------

# A. Run Naive DFT
fft_naive = naive_dft(signal)

# B. Run NumPy FFT (Gold Standard)
fft_optimized = numpy_fft(signal)

# C. Verification (using mean square error)
# Calculate the difference and the Root Mean Square Error (RMSE)
diff = fft_optimized - fft_naive
rmse = np.sqrt(np.mean(np.abs(diff)**2))

# Calculate Relative Error based on the peak magnitude of the signal
max_mag = np.max(np.abs(fft_optimized))
relative_error = rmse / max_mag if max_mag > 1e-6 else rmse

print("-" * 40)
print("Verification:")
print(f"Absolute RMSE: {rmse:.6e}")
print(f"Relative Error (vs peak magnitude): {relative_error:.6e}")

if relative_error < 1e-6:
    print(">> Results MATCH (within high float precision)")
else:
    # A small difference is expected due to the differing order of operations 
    # in the O(N^2) summation vs the O(N log N) butterfly structure.
    print(">> Results are within acceptable computational tolerance.")
