from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

IMAGE_DIR      = r"C:\Users\mu00129\Desktop\slmnew10\capturesCameraSquareNew4"

CAM_ROW_START  = 567
CAM_ROW_END    = 1233

CAM_NOSHIFT_C1 = 1100
CAM_NOSHIFT_C2 = 1300

CAM_SHIFT_C1   = 243
CAM_SHIFT_C2   = 700

KC             = 333



def _find_central_frequency(L_s):
    N = len(L_s)
    mag = np.abs(np.fft.fft(L_s)[:N//2])
    skip = 14
    return (int(np.argmax(mag[skip:])) + skip) / N * 2.0

def find_phi(I, kc=KC, band_width=0.01, env_frac=1/6, mute_first=200):
    I = np.asarray(I, dtype=float).ravel()
    N = I.size;  k = np.arange(N, dtype=float)
    L_s = I - I.mean()
    CF  = _find_central_frequency(L_s)
    Mf  = int(round(N/4));
    if Mf % 2 == 1: Mf += 1
    numtaps = Mf + 1
    eps = np.finfo(float).eps**0.5;  bw = float(band_width)
    CF  = float(np.clip(CF, bw+eps, 1.0-bw-eps))
    b   = signal.firwin(numtaps, [max(CF-bw,eps), min(CF+bw,1-eps)], pass_zero=False)
    L_s = signal.lfilter(b, [1.0], L_s)
    L_a = signal.hilbert(L_s);  nd = Mf//2
    L_s = L_s[nd:];  L_a = L_a[nd:];  k = k[:L_s.size]
    evlp = np.abs(L_a)
    if 0 < mute_first < evlp.size: evlp[:mute_first] = 0.0
    idx = np.where(evlp > evlp.max()*env_frac)[0]
    if idx.size == 0: raise ValueError("No valid data")
    nb = max(idx[0], Mf-nd);  ne = idx[-1]
    L_s=L_s[nb:ne+1]; L_a=L_a[nb:ne+1]; k=k[nb:ne+1]
    a = np.abs(L_a); a[a==0]=1.0; L_s/=a; L_a/=a
    phi_n = np.unwrap(np.arctan2(np.imag(L_a), np.real(L_a))*2.0)/2.0
    if np.real(L_a[0]) < 0: phi_n -= np.pi
    est,*_ = np.linalg.lstsq(np.vstack([k,np.ones_like(k)]).T, phi_n, rcond=None)
    L_est, phi0 = est
    phi0 += 2*np.pi*(-np.round(phi0/(2*np.pi)) - 1.0)
    return float(L_est*kc + phi0)


change = []
for i in range(256):
    s = f"{i:03d}"
    img    = Image.open(f"{IMAGE_DIR}/Capture_gray_{s}.bmp").convert("L")
    matrix = np.array(img)

    NoShift = np.mean(matrix[CAM_ROW_START:CAM_ROW_END,
                              CAM_NOSHIFT_C1:CAM_NOSHIFT_C2], axis=1)
    Shift   = np.mean(matrix[CAM_ROW_START:CAM_ROW_END,
                              CAM_SHIFT_C1:CAM_SHIFT_C2],   axis=1)

    try:
        change.append(find_phi(Shift) - find_phi(NoShift))
    except Exception as e:
        print(f"  gray {i:03d} failed: {e}")
        change.append(np.nan)
    
    if i % 32 == 0:
        print(f"  [{i}/255] done")

change = np.array(change)
valid  = ~np.isnan(change)
change[valid] = np.unwrap(change[valid], period=np.pi)
change -= change[np.where(valid)[0][0]]
if change[np.where(valid)[0][-1]] < 0:
    change = -change
change_pi = change / np.pi


gray_axis = np.arange(256)
ideal     = np.linspace(0, 2, 256)   # 0 to 2π in units of π
deviation = change_pi - ideal
max_dev   = np.nanmax(np.abs(deviation))
print(f"\nMax deviation from ideal: {max_dev:.4f}π")
print(f"Status: {'PASS ✓' if max_dev < 0.1 else 'FAIL ✗'}")



fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(gray_axis, change_pi, label="Measured", linewidth=1)
axes[0].plot(gray_axis, ideal, '--', color='orange', label="Ideal 0→2π")
axes[0].set_xlabel("Gray Scale [8-bit]")
axes[0].set_ylabel("Phase [Rad/π]")
axes[0].set_title("Phase vs Gray Scale")
axes[0].legend()

axes[1].plot(gray_axis, deviation, linewidth=1)
axes[1].axhline( 0.1, color='r', linestyle='--', label='+0.1π limit')
axes[1].axhline(-0.1, color='r', linestyle='--', label='-0.1π limit')
axes[1].set_xlabel("Gray Scale [8-bit]")
axes[1].set_ylabel("Deviation [Rad/π]")
axes[1].set_title(f"Deviation from ideal  (max={max_dev:.4f}π)")
axes[1].legend()

plt.tight_layout()
plt.savefig("linearity_result.png", dpi=150)
plt.show()
