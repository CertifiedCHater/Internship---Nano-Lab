import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
from scipy.stats import norm
 
 
# =============================================================================
# SECTION 1: CONFIGURATION  — edit these before running
# =============================================================================
 
# --- Paths ---
OUTPUT_DIR    = r"C:\path\to\save\captures"   # laptop new file
CALIB_DIR     = OUTPUT_DIR                     # same folder for calibration check
 
# --- SLM hardware settings ---
SLM_WIDTH     = 1920
SLM_HEIGHT    = 1200
 
# --- Rectangle (Shift region) on SLM image ---
# This defines the rectangular region that gets the cycling gray scale.
# Everything outside stays at gray 0 (the NoShift reference).
# Adjust these to match where your beam hits and what size rectangle you want.
RECT_ROW_START = 200     # top of rectangle on SLM (pixels)
RECT_ROW_END   = 1000    # bottom of rectangle on SLM (pixels)
RECT_COL_START = 700     # left edge of rectangle on SLM (pixels)
RECT_COL_END   = 1200    # right edge of rectangle on SLM (pixels)
 
# --- Camera ROI for phase extraction ---
# These are the row/col ranges used to extract the 1D intensity profiles
# from the CAMERA image (not the SLM image).
# Based on your image analysis:
#   NoShift region (left of rectangle): cols 200-650
#   Shift region (inside rectangle):    cols 715-1135
#   Row range (where fringes are clean): rows 600-1400
CAM_ROW_START   = 600
CAM_ROW_END     = 1400
CAM_NOSHIFT_C1  = 200    # NoShift column start
CAM_NOSHIFT_C2  = 650    # NoShift column end
CAM_SHIFT_C1    = 750    # Shift column start
CAM_SHIFT_C2    = 1100   # Shift column end
 
# --- Phase extraction settings ---
# kc: the fixed row index where phase is evaluated on the vertical profile
# Must be within [CAM_ROW_START, CAM_ROW_END] and the same for ALL images
KC              = 900    # row index for phase evaluation
 
# --- Sweep settings ---
NUM_GRAY_LEVELS = 256
CALIB_PREFIX    = "Capture_gray_"
CALIB_SUFFIX    = ".bmp"
CALIB_RANGE_START = 0
CALIB_RANGE_END   = 255
 
# --- Holoeye SDK path (adjust if different on your machine) ---
HOLOEYE_SDK_PATH = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.0.0\examples"
 
 
# =============================================================================
# SECTION 2: HELPER FUNCTIONS
# =============================================================================
 
def find_central_frequency(L_s):
    N = len(L_s)
    FFTLs = np.fft.fft(L_s)
    half  = np.abs(FFTLs[:N // 2])
    skip  = 14
    if half.size <= skip:
        raise ValueError("Signal too short.")
    yy = int(np.argmax(half[skip:]))
    return (yy + skip) / N * 2.0
 
 
def find_phi(I, kc=KC, band_width=0.01, env_frac=1/6, mute_first=200):
    """
    Extract phase from a 1D interferometric intensity profile.
    Based on Shen & Wang (2005) / Ma & Wang (2013)
 
    Parameters
    ----------
    I          : 1D intensity array (vertical profile from one camera region)
    kc         : fixed row index where phase is evaluated — MUST be same for all images
    band_width : FIR filter half-bandwidth (normalised)
    env_frac   : envelope threshold fraction
    mute_first : samples to ignore at start (suppress edge effects)
    """
    I  = np.asarray(I, dtype=float).ravel()
    N  = I.size
    k  = np.arange(N, dtype=float)
    L_s = I - I.mean()
 
    CF  = find_central_frequency(L_s)
    Mf  = int(round(N / 4))
    if Mf % 2 == 1:
        Mf += 1
    numtaps = Mf + 1
 
    eps  = np.finfo(float).eps ** 0.5
    bw   = float(band_width)
    CF   = float(np.clip(CF, bw + eps, 1.0 - bw - eps))
    low  = max(CF - bw, eps)
    high = min(CF + bw, 1.0 - eps)
 
    b   = signal.firwin(numtaps, [low, high], pass_zero=False)
    L_s = signal.lfilter(b, [1.0], L_s)
    L_a = signal.hilbert(L_s)
    nd  = Mf // 2
 
    L_s = L_s[nd:];  L_a = L_a[nd:];  k = k[:L_s.size]
 
    evlp = np.abs(L_a)
    if 0 < mute_first < evlp.size:
        evlp[:mute_first] = 0.0
    idx_eff = np.where(evlp > evlp.max() * env_frac)[0]
    if idx_eff.size == 0:
        raise ValueError("No valid data in phase extraction.")
 
    N_begin = max(idx_eff[0], Mf - nd)
    N_end   = idx_eff[-1]
    L_s = L_s[N_begin:N_end+1];  L_a = L_a[N_begin:N_end+1];  k = k[N_begin:N_end+1]
 
    abs_La = np.abs(L_a);  abs_La[abs_La == 0] = 1.0
    L_s = L_s / abs_La;    L_a = L_a / abs_La
 
    phi_n = np.unwrap(np.arctan2(np.imag(L_a), np.real(L_a)) * 2.0) / 2.0
    if np.real(L_a[0]) < 0:
        phi_n = phi_n - np.pi
 
    A = np.vstack([k, np.ones_like(k)]).T
    est, *_ = np.linalg.lstsq(A, phi_n, rcond=None)
    L_est, phi0_est1 = est
 
    Nshift    = -np.round(phi0_est1 / (2 * np.pi))
    phi0_est2 = phi0_est1 + 2 * np.pi * (Nshift - 1.0)
 
    return float(L_est * kc + phi0_est2)
 
 
def get_profiles(image_array):
    """
    Extract NoShift and Shift vertical intensity profiles from a camera image.
 
    Your fringes are horizontal so we read them with a vertical profile.
    We average horizontally over a column band to reduce noise.
 
    Returns
    -------
    noshift : 1D array — vertical profile from the reference (left) region
    shift   : 1D array — vertical profile from the rectangle (center) region
    """
    noshift = np.mean(image_array[CAM_ROW_START:CAM_ROW_END,
                                   CAM_NOSHIFT_C1:CAM_NOSHIFT_C2], axis=1)
    shift   = np.mean(image_array[CAM_ROW_START:CAM_ROW_END,
                                   CAM_SHIFT_C1:CAM_SHIFT_C2],   axis=1)
    return noshift, shift
 
 
def build_slm_pattern(gray_val):
    """
    Build an SLM image where:
      - The rectangle region gets gray_val (the phase being tested)
      - Everything else stays at 0 (reference)
 
    Parameters
    ----------
    gray_val : int 0-255
 
    Returns
    -------
    img_array : uint8 ndarray shape (SLM_HEIGHT, SLM_WIDTH)
    """
    img_array = np.zeros((SLM_HEIGHT, SLM_WIDTH), dtype=np.uint8)
    img_array[RECT_ROW_START:RECT_ROW_END,
              RECT_COL_START:RECT_COL_END] = gray_val
    return img_array
 
 
# =============================================================================
# SECTION 3: LIVE CAPTURE — sweep gray 0-255 and save images
# =============================================================================
 
def run_capture(output_dir=OUTPUT_DIR):
    """
    Display each gray scale on SLM and capture the resulting interference image.
    Saves 256 BMP files named Capture_gray_000.bmp ... Capture_gray_255.bmp
 
    Requires HOLOEYE SDK and PySpin to be installed on the lab laptop.
    """
    print("=" * 60)
    print("SECTION 3: LIVE CAPTURE")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)
 
    # --- Load HOLOEYE SDK ---
    sys.path.append(HOLOEYE_SDK_PATH)
    try:
        import HEDS
        from hedslib.heds_types import HEDSERR_NoError
        err = HEDS.SDK.Init(4, 0)
        assert err == HEDSERR_NoError
        slm = HEDS.SLM.Init()
        assert slm.errorCode() == HEDSERR_NoError
        print("  SLM initialized.")
    except Exception as e:
        print(f"  ERROR: SLM init failed: {e}")
        return
 
    # --- Load PySpin (Spinnaker) camera ---
    try:
        import PySpin
        system = PySpin.System.GetInstance()
        cams   = system.GetCameras()
        assert cams.GetSize() > 0, "No camera found"
        camera = cams.GetByIndex(0)
        camera.Init()
 
        # Set continuous acquisition mode
        nodemap = camera.GetNodeMap()
        acq     = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        acq.SetIntValue(acq.GetEntryByName("Continuous").GetValue())
        camera.BeginAcquisition()
 
        # Discard first 20 frames — camera sends duplicate frames at startup
        # (see report section 7.1.2 — first 10 frames are identical)
        print("  Warming up camera (discarding first 20 frames)...")
        for _ in range(20):
            img = camera.GetNextImage()
            img.Release()
        print("  Camera ready.")
    except Exception as e:
        print(f"  ERROR: Camera init failed: {e}")
        slm.close()
        HEDS.SDK.Close()
        return
 
    # --- Sweep ---
    print(f"\n  Capturing {NUM_GRAY_LEVELS} gray levels...")
    for gray_val in range(NUM_GRAY_LEVELS):
        img_array = build_slm_pattern(gray_val)
 
        err, handle = slm.loadImageData(img_array)
        assert err == HEDSERR_NoError
        err = handle.show()
        assert err == HEDSERR_NoError
        time.sleep(0.1)     # wait for SLM to settle
 
        try:
            raw = camera.GetNextImage()
            if not raw.IsIncomplete():
                frame    = raw.GetNDArray().astype(np.uint8)
                filename = f"{CALIB_PREFIX}{gray_val:03d}{CALIB_SUFFIX}"
                Image.fromarray(frame).save(os.path.join(output_dir, filename))
            raw.Release()
        except Exception as e:
            print(f"  WARNING: Capture failed at gray {gray_val}: {e}")
 
        if gray_val % 32 == 0:
            print(f"  {gray_val}/255 done")
 
    # --- Cleanup ---
    camera.EndAcquisition()
    camera.DeInit()
    del camera
    cams.Clear()
    system.ReleaseInstance()
    slm.close()
    HEDS.SDK.Close()
    print(f"\n  Capture complete. Images saved to: {output_dir}")
 
 
# =============================================================================
# SECTION 4: CALIBRATION CHECK — process saved images and plot phase curve
# =============================================================================
 
def run_calibration_check(calib_dir=CALIB_DIR):
    """
    Load saved BMP images, extract phase at each gray scale using
    
    The subtraction cancels out mechanical vibration and laser drift
    since both regions see the same global phase noise.
    """
    print("=" * 60)
    print("SECTION 4: CALIBRATION CHECK")
    print("=" * 60)
    print(f"  Loading from : {calib_dir}")
    print(f"  Gray range   : {CALIB_RANGE_START} to {CALIB_RANGE_END}")
    print(f"  NoShift cols : {CAM_NOSHIFT_C1} to {CAM_NOSHIFT_C2}")
    print(f"  Shift cols   : {CAM_SHIFT_C1} to {CAM_SHIFT_C2}")
    print(f"  Row range    : {CAM_ROW_START} to {CAM_ROW_END}")
    print(f"  kc           : {KC}\n")
 
    gray_range = range(CALIB_RANGE_START, CALIB_RANGE_END + 1)
    phase_diffs = []
    missing     = []
 
    for i in gray_range:
        fname = f"{CALIB_PREFIX}{i:03d}{CALIB_SUFFIX}"
        fpath = os.path.join(calib_dir, fname)
 
        if not os.path.exists(fpath):
            print(f"  WARNING: Missing {fname}")
            missing.append(i)
            phase_diffs.append(np.nan)
            continue
 
        arr = np.array(Image.open(fpath).convert("L"), dtype=float)
        noshift, shift = get_profiles(arr)
 
        try:
            phi_ns  = find_phi(noshift, kc=KC)
            phi_s   = find_phi(shift,   kc=KC)
            phase_diffs.append(phi_s - phi_ns)
        except Exception as e:
            print(f"  WARNING: Phase extraction failed at gray {i:03d}: {e}")
            phase_diffs.append(np.nan)
 
    # --- Post-processing ---
    change = np.array(phase_diffs)
 
    # Unwrap to remove 2π jumps
    valid_mask = ~np.isnan(change)
    change[valid_mask] = np.unwrap(change[valid_mask], period=np.pi)
 
    # Start from zero
    first_valid = np.where(valid_mask)[0][0]
    change -= change[first_valid]
 
    # Ensure positive slope
    last_valid = np.where(valid_mask)[0][-1]
    if change[last_valid] < 0:
        change = -change
 
    # Normalize to π units
    change_pi = change / np.pi
 
    # Ideal linear reference
    gray_axis = np.array(list(gray_range))
    ideal     = np.linspace(0, 2, len(gray_range))
 
    # Deviation from ideal
    deviation = change_pi - ideal
    max_dev   = np.nanmax(np.abs(deviation))
 
    print(f"  Max phase deviation : {max_dev:.4f}π")
    print(f"  Requirement         : < 0.1π")
    print(f"  Status              : {'PASS ✓' if max_dev < 0.1 else 'FAIL ✗'}")
    if missing:
        print(f"  Missing files       : {missing}")
 
    # --- Plot: replicate figure 17 from report ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("SLM Calibration Check", fontsize=13)
 
    # Left: phase vs gray scale
    axes[0].plot(gray_axis, change_pi, label="Measured phase", linewidth=1)
    axes[0].plot(gray_axis, ideal,     label="Ideal linear",
                 linestyle="--", color="orange")
    axes[0].set_xlabel("Gray Scale [8-bit]")
    axes[0].set_ylabel("Phase Shift [Rad/π]")
    axes[0].set_title("Phase vs Gray Scale")
    axes[0].legend()
 
    # Right: deviation from ideal
    axes[1].plot(gray_axis, deviation, linewidth=1)
    axes[1].axhline( 0.1, color="r", linestyle="--", label="+0.1π limit")
    axes[1].axhline(-0.1, color="r", linestyle="--", label="-0.1π limit")
    axes[1].set_xlabel("Gray Scale [8-bit]")
    axes[1].set_ylabel("Deviation [Rad/π]")
    axes[1].set_title(f"Phase Deviation from Ideal  (max = {max_dev:.4f}π)")
    axes[1].legend()
 
    plt.tight_layout()
    out_path = os.path.join(calib_dir, "calibration_result.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\n  Plot saved to: {out_path}")
 
    return change_pi, deviation, max_dev
 
 
# =============================================================================
# SECTION 5: FIGURE 11 REPLICA — show camera image + profiles side by side
# =============================================================================
 
def show_figure11(image_path_ref, image_path_shifted=None):
    """
    Reproduce figure 11 from the report:
      Left  — camera image with the NoShift and Shift ROI rectangles drawn on it
      Right — the two 1D intensity profiles overlaid
 
    Parameters
    ----------
    image_path_ref     : path to one BMP (e.g. gray_000 — the reference)
    image_path_shifted : path to another BMP at a different gray level (optional)
                         if provided, both profiles are plotted together
    """
    arr_ref = np.array(Image.open(image_path_ref).convert("L"), dtype=float)
    noshift_ref, shift_ref = get_profiles(arr_ref)
    rows = np.arange(CAM_ROW_START, CAM_ROW_END)
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 11 replica — camera image + interference profiles", fontsize=12)
 
    # Left: camera image with ROI rectangles
    axes[0].imshow(arr_ref, cmap="gray", vmin=0, vmax=255)
    from matplotlib.patches import Rectangle
    # NoShift rectangle (blue)
    axes[0].add_patch(Rectangle(
        (CAM_NOSHIFT_C1, CAM_ROW_START),
        CAM_NOSHIFT_C2 - CAM_NOSHIFT_C1,
        CAM_ROW_END - CAM_ROW_START,
        linewidth=2, edgecolor="#378ADD", facecolor="none", label="NoShift"
    ))
    # Shift rectangle (red)
    axes[0].add_patch(Rectangle(
        (CAM_SHIFT_C1, CAM_ROW_START),
        CAM_SHIFT_C2 - CAM_SHIFT_C1,
        CAM_ROW_END - CAM_ROW_START,
        linewidth=2, edgecolor="#E24B4A", facecolor="none", label="Shift"
    ))
    axes[0].set_title("Camera image + ROI rectangles")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].axis("off")
 
    # Right: intensity profiles
    axes[1].plot(rows, noshift_ref, color="#378ADD", linewidth=1,
                 label=f"NoShift — {os.path.basename(image_path_ref)}")
    axes[1].plot(rows, shift_ref,   color="#E24B4A", linewidth=1,
                 label=f"Shift — {os.path.basename(image_path_ref)}")
 
    if image_path_shifted:
        arr_s = np.array(Image.open(image_path_shifted).convert("L"), dtype=float)
        noshift_s, shift_s = get_profiles(arr_s)
        axes[1].plot(rows, noshift_s, color="#378ADD", linewidth=1,
                     linestyle="--", alpha=0.6,
                     label=f"NoShift — {os.path.basename(image_path_shifted)}")
        axes[1].plot(rows, shift_s,   color="#E24B4A", linewidth=1,
                     linestyle="--", alpha=0.6,
                     label=f"Shift — {os.path.basename(image_path_shifted)}")
 
    axes[1].set_xlabel("Row (y)")
    axes[1].set_ylabel("Normalised amplitude")
    axes[1].set_title("Averaged interference fringes on camera")
    axes[1].legend(fontsize=8)
 
    plt.tight_layout()
    plt.savefig("figure11_replica.png", dpi=150)
    plt.show()
    print("  Saved figure11_replica.png")
 
 
# =============================================================================
# MAIN
# =============================================================================
 
if __name__ == "__main__":
 
    # --- Step 1: Capture images (run with SLM + camera connected) ---
    # Uncomment when ready to run live in the lab
    # run_capture(OUTPUT_DIR)
 
    # --- Step 2: Check calibration from saved BMPs ---
    # Point this at your folder of 256 BMP files
    run_calibration_check(CALIB_DIR)
 
    # --- Step 3: Show figure 11 replica for two gray levels ---
    # show_figure11(
    #     r"C:\path\to\Capture_gray_000.bmp",
    #     r"C:\path\to\Capture_gray_128.bmp"
    # )
