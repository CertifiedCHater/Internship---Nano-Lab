import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal


OUTPUT_DIR = r"C:\Users\mu00129\Desktop\slmnew10\Measurements2\PLUTO\calib_single"
HOLOEYE_SDK_PATH = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.0.0\examples"

SLM_WIDTH, SLM_HEIGHT = 1920, 1080     
NUM_GRAY_LEVELS = 256
SPLIT_COL = None                        
                                       

LASER_WAVELENGTH = 633.0
EXPOSURE_US   = 85.0
WARMUP_FRAMES = 20                      
SETTLE_TIME   = 1.5                     
FLUSH_FRAMES  = 3                       


CAM_ROW_START, CAM_ROW_END     = 700, 1400
CAM_NOSHIFT_C1, CAM_NOSHIFT_C2 = 300, 700
CAM_SHIFT_C1, CAM_SHIFT_C2     = 1200, 1600
KC = 362

DO_CAPTURE  = True                      
DO_ANALYZE  = True
TEMP_BMP = os.path.join(OUTPUT_DIR, "_temp_pattern.bmp")


def _find_central_frequency(L_s):
    N = len(L_s)
    mag = np.abs(np.fft.fft(L_s)[:N // 2])
    skip = 14
    if mag.size <= skip:
        raise ValueError("Signal too short to estimate central frequency.")
    yy = int(np.argmax(mag[skip:]))
    return (yy + skip) / N * 2.0


def find_phi(I, kc=362.0, band_width=0.01, env_frac=1/6, mute_first=200):
    I = np.asarray(I, dtype=float).ravel()
    N = I.size
    k = np.arange(N, dtype=float)
    L_s = I - I.mean()
    CF = _find_central_frequency(L_s)
    Mf = int(round(N / 4))
    if Mf % 2 == 1:
        Mf += 1
    numtaps = Mf + 1
    eps = np.finfo(float).eps ** 0.5
    bw = float(band_width)
    CF = float(np.clip(CF, bw + eps, 1.0 - bw - eps))
    low = max(CF - bw, eps)
    high = min(CF + bw, 1.0 - eps)
    if not (low < high):
        raise ValueError("Invalid bandpass range; decrease band_width or check CF.")
    b = signal.firwin(numtaps, [low, high], pass_zero=False)
    L_s = signal.lfilter(b, [1.0], L_s)
    L_a = signal.hilbert(L_s)
    nd = Mf // 2
    L_s = L_s[nd:]
    L_a = L_a[nd:]
    k = k[:L_s.size]
    evlp = np.abs(L_a)
    if 0 < mute_first < evlp.size:
        evlp[:mute_first] = 0.0
    idx_eff = np.where(evlp > evlp.max() * env_frac)[0]
    if idx_eff.size == 0:
        raise ValueError("No valid data found; adjust env_frac/mute_first.")
    N_begin = max(idx_eff[0], Mf - nd)
    N_end = idx_eff[-1]
    L_s = L_s[N_begin:N_end + 1]
    L_a = L_a[N_begin:N_end + 1]
    k = k[N_begin:N_end + 1]
    abs_La = np.abs(L_a)
    abs_La[abs_La == 0] = 1.0
    L_s = L_s / abs_La
    L_a = L_a / abs_La
    phi_n = np.unwrap(np.arctan2(np.imag(L_a), np.real(L_a)) * 2.0) / 2.0
    if np.real(L_a[0]) < 0:
        phi_n = phi_n - np.pi
    A = np.vstack([k, np.ones_like(k)]).T
    est, *_ = np.linalg.lstsq(A, phi_n, rcond=None)
    L_est, phi0_est1 = est
    Nshift = -np.round(phi0_est1 / (2 * np.pi))
    phi0_est2 = phi0_est1 + 2 * np.pi * (Nshift - 1.0)
    return float(L_est * kc + phi0_est2)



def build_pattern(gray_val):
    """Half-and-half: left half at gray 0 (reference), right half at gray_val."""
    img = np.zeros((SLM_HEIGHT, SLM_WIDTH), dtype=np.uint8)
    col = SLM_WIDTH // 2 if SPLIT_COL is None else SPLIT_COL
    img[:, col:] = gray_val
    return img



def run_capture():
    global SLM_WIDTH, SLM_HEIGHT
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sys.path.append(HOLOEYE_SDK_PATH)
    import HEDS
    from hedslib.heds_types import HEDSERR_NoError

    assert HEDS.SDK.Init(4, 0) == HEDSERR_NoError, "SDK Init failed"
    slm = HEDS.SLM.Init("", True, 0.0)                 # PLUTO init
    assert slm.errorCode() == HEDSERR_NoError, "SLM Init failed"
    slm.setWavelength(LASER_WAVELENGTH)                # PLUTO needs this
    SLM_WIDTH, SLM_HEIGHT = slm.width_px(), slm.height_px()   # use the real size
    print(f"  SLM ready: {SLM_WIDTH} x {SLM_HEIGHT} px")

    import PySpin
    system = PySpin.System.GetInstance()
    cams = system.GetCameras()
    assert cams.GetSize() > 0, "No camera found"
    camera = cams.GetByIndex(0)
    camera.Init()
    nm = camera.GetNodeMap()
    ea = PySpin.CEnumerationPtr(nm.GetNode("ExposureAuto"))    # manual exposure
    ea.SetIntValue(ea.GetEntryByName("Off").GetValue())
    PySpin.CFloatPtr(nm.GetNode("ExposureTime")).SetValue(EXPOSURE_US)
    ac = PySpin.CEnumerationPtr(nm.GetNode("AcquisitionMode"))
    ac.SetIntValue(ac.GetEntryByName("Continuous").GetValue())
    camera.BeginAcquisition()
    for _ in range(WARMUP_FRAMES):
        camera.GetNextImage().Release()
    print("  Camera ready.")

    print(f"  Capturing {NUM_GRAY_LEVELS} gray levels...")
    for gray_val in range(NUM_GRAY_LEVELS):
        Image.fromarray(build_pattern(gray_val)).save(TEMP_BMP)
        err, handle = slm.loadPhaseDataFromFile(TEMP_BMP)      # PLUTO: phase-data path
        if err != HEDSERR_NoError:
            print(f"  load failed at gray {gray_val}: {err}")
            continue
        handle.show()
        time.sleep(SETTLE_TIME)
        for _ in range(FLUSH_FRAMES):                          # PLUTO: flush stale frames
            camera.GetNextImage().Release()
        try:
            raw = camera.GetNextImage()
            if not raw.IsIncomplete():
                frame = raw.GetNDArray().astype(np.uint8)
                Image.fromarray(frame).save(
                    os.path.join(OUTPUT_DIR, f"Capture_gray_{gray_val:03d}.bmp"))
            raw.Release()
        except Exception as e:
            print(f"  capture failed at gray {gray_val}: {e}")
        if gray_val % 32 == 0 or gray_val == 255:
            print(f"  [{gray_val:3d}/255]")

    try:
        camera.EndAcquisition(); camera.DeInit(); del camera
        cams.Clear(); system.ReleaseInstance()
    except Exception:
        pass
    try:
        HEDS.SDK.Close()                                       # no slm.close() in SDK 4.0
    except Exception:
        pass
    if os.path.exists(TEMP_BMP):
        os.remove(TEMP_BMP)
    print("  Capture complete ->", OUTPUT_DIR)


def run_analysis():
    change = []
    grays = []
    for g in range(NUM_GRAY_LEVELS):
        path = os.path.join(OUTPUT_DIR, f"Capture_gray_{g:03d}.bmp")
        if not os.path.exists(path):
            continue
        matrix = np.array(Image.open(path).convert("L"), dtype=float)
        NoShift = matrix[CAM_ROW_START:CAM_ROW_END, CAM_NOSHIFT_C1:CAM_NOSHIFT_C2].mean(axis=1)
        Shift   = matrix[CAM_ROW_START:CAM_ROW_END, CAM_SHIFT_C1:CAM_SHIFT_C2].mean(axis=1)
        try:
            change.append(find_phi(Shift, kc=KC) - find_phi(NoShift, kc=KC))
            grays.append(g)
        except Exception as e:
            print(f"  phase extraction failed at gray {g}: {e}")

    if len(change) < 2:
        print("  not enough valid frames to analyse.")
        return
    change = np.unwrap(np.array(change), period=np.pi) / np.pi   # stitch pi-ambiguity
    change = change - change.min()
    depth = change.max() - change.min()
    print(f"\n  modulation depth = {depth:.3f} pi   (need >= 2 pi to calibrate)")

    plt.figure(figsize=(8, 5))
    plt.plot(grays, change, ".-")
    plt.axhline(2.0, color="r", ls="--", label="2π target")
    plt.xlabel("gray value"); plt.ylabel("phase shift [π]")
    plt.title(f"PLUTO phase vs gray — depth = {depth:.3f} π")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "pluto_phase_curve.png")
    plt.savefig(out, dpi=140); plt.show()
    print("  saved", out)

if __name__ == "__main__":
    if DO_CAPTURE:
        run_capture()
    if DO_ANALYZE:
        run_analysis()
