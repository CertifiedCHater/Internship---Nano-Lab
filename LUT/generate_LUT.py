import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm



IMAGE_DIR   = r"C:\Users\mu00129\Desktop\slmnew10\capturesCameraSquareNew5"
OUTPUT_DIR  = r"C:\Users\mu00129\Desktop\slmnew10\CalibrationFiles"


CURRENT_LUT = r"C:\path\to\ERISCalibration_NIR-153_635.00nm_2.30pi_sgl=2100.hecalib.txt"


ITERATION   = 1
OUTPUT_NAME = f"Calibration_Iterative_{ITERATION}"


CAM_ROW_START  = 447
CAM_ROW_END    = 1540
CAM_NOSHIFT_C1 = 1300
CAM_NOSHIFT_C2 = 1600
CAM_SHIFT_C1   = 400
CAM_SHIFT_C2   = 900
KC             = 546

CALIB_PREFIX = "Capture_gray_"
CALIB_SUFFIX = ".bmp"


CORRECTION_DIVISOR = 5



def _find_central_frequency(L_s):
    N    = len(L_s)
    mag  = np.abs(np.fft.fft(L_s)[:N//2])
    skip = 14
    return (int(np.argmax(mag[skip:])) + skip) / N * 2.0


def find_phi(I, kc=KC, band_width=0.01, env_frac=1/6, mute_first=200):
    I   = np.asarray(I, dtype=float).ravel()
    N   = I.size
    k   = np.arange(N, dtype=float)
    L_s = I - I.mean()
    CF  = _find_central_frequency(L_s)
    Mf  = int(round(N / 4))
    if Mf % 2 == 1: Mf += 1
    numtaps = Mf + 1
    eps  = np.finfo(float).eps**0.5
    bw   = float(band_width)
    CF   = float(np.clip(CF, bw+eps, 1.0-bw-eps))
    b    = signal.firwin(numtaps, [max(CF-bw,eps), min(CF+bw,1-eps)], pass_zero=False)
    L_s  = signal.lfilter(b, [1.0], L_s)
    L_a  = signal.hilbert(L_s)
    nd   = Mf//2
    L_s  = L_s[nd:];  L_a = L_a[nd:];  k = k[:L_s.size]
    evlp = np.abs(L_a)
    if 0 < mute_first < evlp.size: evlp[:mute_first] = 0.0
    idx  = np.where(evlp > evlp.max()*env_frac)[0]
    if idx.size == 0: raise ValueError("No valid data.")
    nb = max(idx[0], Mf-nd);  ne = idx[-1]
    L_s=L_s[nb:ne+1]; L_a=L_a[nb:ne+1]; k=k[nb:ne+1]
    a = np.abs(L_a); a[a==0]=1.0; L_s/=a; L_a/=a
    phi_n = np.unwrap(np.arctan2(np.imag(L_a), np.real(L_a))*2.0)/2.0
    if np.real(L_a[0]) < 0: phi_n -= np.pi
    est,*_ = np.linalg.lstsq(np.vstack([k,np.ones_like(k)]).T, phi_n, rcond=None)
    L_est, phi0 = est
    phi0 += 2*np.pi*(-np.round(phi0/(2*np.pi)) - 1.0)
    return float(L_est*kc + phi0)


def get_profiles(arr):
    ns = np.mean(arr[CAM_ROW_START:CAM_ROW_END, CAM_NOSHIFT_C1:CAM_NOSHIFT_C2], axis=1)
    sh = np.mean(arr[CAM_ROW_START:CAM_ROW_END, CAM_SHIFT_C1:CAM_SHIFT_C2],   axis=1)
    return ns, sh




def resample_to_10bit(y8):
    x8     = np.linspace(0, 255, 256)
    f      = PchipInterpolator(x8, np.asarray(y8, float), extrapolate=False)
    x10    = np.arange(1024, dtype=float)
    x8_eval = x10 * (255.0 / 1023.0)
    return f(x8_eval)




def generate_lut(image_dir=IMAGE_DIR, current_lut=CURRENT_LUT,
                 output_dir=OUTPUT_DIR, iteration=ITERATION):

    os.makedirs(output_dir, exist_ok=True)

    print("Extracting phase from 256 images...")
    phase_diffs = []

    for gv in range(256):
        fname = os.path.join(image_dir, f"{CALIB_PREFIX}{gv:03d}{CALIB_SUFFIX}")
        if not os.path.exists(fname):
            print(f"  Missing: {fname}")
            phase_diffs.append(np.nan)
            continue

        arr = np.array(Image.open(fname).convert("L"), dtype=float)
        ns, sh = get_profiles(arr)

        try:
            phase_diffs.append(find_phi(sh) - find_phi(ns))
        except Exception as e:
            print(f"  gray {gv:03d} failed: {e}")
            phase_diffs.append(np.nan)

        if gv % 32 == 0:
            print(f"  [{gv}/255] done")


    change     = np.array(phase_diffs)
    valid      = ~np.isnan(change)
    change[valid] = np.unwrap(change[valid], period=np.pi)
    change    -= change[np.where(valid)[0][0]]
    if change[np.where(valid)[0][-1]] < 0:
        change = -change

    change_pi = change / np.pi

    # Ideal 0→2 ramp
    gray_axis = np.arange(256)
    ideal     = np.linspace(0, 2, 256)


    difference = change_pi - ideal

    print(f"\nPhase deviation stats:")
    print(f"  Mean : {np.nanmean(difference):.4f}π")
    print(f"  Max  : {np.nanmax(difference):.4f}π")
    print(f"  Min  : {np.nanmin(difference):.4f}π")

    print(f"\nLoading current LUT: {current_lut}")
    try:
        with open(current_lut, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # LUT values start at line 9 (after the 8-line header)
        codes_10bit = np.array([int(h, 16) for h in lines[9:]], dtype=np.int32)
        print(f"  Loaded {len(codes_10bit)} LUT entries")
        print(f"  LUT range: {codes_10bit.min()} to {codes_10bit.max()}")
    except FileNotFoundError:
        print(f"  ERROR: LUT file not found: {current_lut}")
        print("  Using flat linear LUT as starting point instead.")
        codes_10bit = np.linspace(0, 4095, 1024).astype(np.int32)

    diff_10bit = resample_to_10bit(difference)


    correction = codes_10bit - diff_10bit / (2 * np.pi) * 4096 / CORRECTION_DIVISOR

    correction = np.clip(correction, 0, 4095)

    x       = np.arange(1024)
    coeffs  = np.polyfit(x, correction, 5)
    smoothed = np.polyval(coeffs, x)
    smoothed = np.clip(smoothed, 0, 4095)

    header_lines = lines[:9]

    raw_path    = os.path.join(output_dir, f"{OUTPUT_NAME}.hecalib.txt")
    smooth_path = os.path.join(output_dir, f"{OUTPUT_NAME}_Smooth.hecalib.txt")

    for path, data, label in [
        (raw_path,    correction, "raw"),
        (smooth_path, smoothed,   "smoothed"),
    ]:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(header_lines)
            for val in data:
                f.write(f"{round(val):03X}\n")
        print(f"  Saved {label} LUT: {path}")

    print(f"\n  → Load '{smooth_path}' into Holoeye SDK")
    print(f"  → Recapture 256 images and run this script again with ITERATION={iteration+1}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"LUT Generation — Iteration {iteration}", fontsize=13)

    axes[0].plot(gray_axis, change_pi, label="Measured", linewidth=1)
    axes[0].plot(gray_axis, ideal, '--', color='orange', label="Ideal 0→2π")
    axes[0].set_xlabel("Gray Scale [8-bit]")
    axes[0].set_ylabel("Phase [Rad/π]")
    axes[0].set_title("Phase vs Gray Scale")
    axes[0].legend()

    axes[1].plot(gray_axis, difference, linewidth=1)
    axes[1].axhline( 0.1, color='r', linestyle='--', label='+0.1π')
    axes[1].axhline(-0.1, color='r', linestyle='--', label='-0.1π')
    axes[1].axhline(np.nanmax(difference), color='g', linestyle='--',
                    label=f"max={np.nanmax(difference):.3f}π")
    axes[1].axhline(np.nanmean(difference), color='b', linestyle='--',
                    label=f"mean={np.nanmean(difference):.3f}π")
    axes[1].set_xlabel("Gray Scale [8-bit]")
    axes[1].set_ylabel("Deviation [Rad/π]")
    axes[1].set_title("Deviation from ideal")
    axes[1].legend(fontsize=8)

    axes[2].plot(codes_10bit, label="Current LUT", linewidth=1)
    axes[2].plot(correction,  label="Corrected (raw)", linewidth=1)
    axes[2].plot(smoothed,    label="Corrected (smooth)", linewidth=1)
    axes[2].set_xlabel("Gray Scale [10-bit]")
    axes[2].set_ylabel("DAC value")
    axes[2].set_title("LUT comparison")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{OUTPUT_NAME}_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"  Plot saved: {plot_path}")

    return change_pi, difference, smoothed


if __name__ == "__main__":
    generate_lut()
