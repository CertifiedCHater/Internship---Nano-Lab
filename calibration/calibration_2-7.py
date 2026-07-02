from __future__ import annotations
import argparse, os, sys, time
import numpy as np
from PIL import Image
from scipy import signal

MODE = "live"                 # "live" | "sweep" | "analyze"

DIR        = r"C:\Users\mu00129\Desktop\slmnew10\Measurements2\PLUTO\depth"
SLM_WIDTH, SLM_HEIGHT = 1920, 1200
RECT_ROW_START, RECT_ROW_END = 400, 680
RECT_COL_START, RECT_COL_END = 150, 450     # square inside the beam footprint

CAM_ROW_START, CAM_ROW_END   = 266, 1130
CAM_NOSHIFT_C1, CAM_NOSHIFT_C2 = 33, 449
CAM_SHIFT_C1, CAM_SHIFT_C2     = 549, 1310
KC = 432

HOLOEYE_SDK_PATH = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.0.0\examples"
LASER_WAVELENGTH = 633.0
USE_PHASE_DATA   = True
EXPOSURE_US      = 85.0
WARMUP_FRAMES    = 20
SETTLE_TIME      = 1.0        # per pattern; live mode can be snappier
FLUSH_FRAMES     = 3
SWEEP_GRAYS      = list(range(0, 256, 4))    # dense sampling so unwrapping works
                                             # (use range(0,256) for the real calibration)
TARGET_PI        = 2.0       # calibration needs the depth to reach this
TEMP_BMP = os.path.join(DIR, "_tmp.bmp")


def find_central_frequency(L_s):
    N = len(L_s); half = np.abs(np.fft.fft(L_s)[:N // 2]); skip = 14
    if half.size <= skip:
        raise ValueError("Signal too short.")
    return (int(np.argmax(half[skip:])) + skip) / N * 2.0

def find_phi(I, kc=KC, band_width=0.015, env_frac=1/6, mute_first=200):
    I = np.asarray(I, dtype=float).ravel(); N = I.size; k = np.arange(N, dtype=float)
    L_s = I - I.mean(); CF = find_central_frequency(L_s)
    Mf = int(round(N / 4));  Mf += Mf % 2; numtaps = Mf + 1
    eps = np.finfo(float).eps ** 0.5; bw = float(band_width)
    CF = float(np.clip(CF, bw + eps, 1.0 - bw - eps))
    low = max(CF - bw, eps); high = min(CF + bw, 1.0 - eps)
    b = signal.firwin(numtaps, [low, high], pass_zero=False)
    L_s = signal.lfilter(b, [1.0], L_s); L_a = signal.hilbert(L_s)
    nd = Mf // 2; L_s = L_s[nd:]; L_a = L_a[nd:]; k = k[:L_s.size]
    evlp = np.abs(L_a)
    if 0 < mute_first < evlp.size: evlp[:mute_first] = 0.0
    idx = np.where(evlp > evlp.max() * env_frac)[0]
    if idx.size == 0: raise ValueError("No valid data.")
    nb = max(idx[0], Mf - nd); ne = idx[-1]
    L_a = L_a[nb:ne + 1]; k = k[nb:ne + 1]
    aL = np.abs(L_a); aL[aL == 0] = 1.0; L_a = L_a / aL
    phi = np.unwrap(np.arctan2(np.imag(L_a), np.real(L_a)) * 2.0) / 2.0
    if np.real(L_a[0]) < 0: phi = phi - np.pi
    A = np.vstack([k, np.ones_like(k)]).T
    (L_est, phi0), *_ = [np.linalg.lstsq(A, phi, rcond=None)[0]]
    phi0 += 2 * np.pi * (-np.round(phi0 / (2 * np.pi)) - 1.0)
    return float(L_est * kc + phi0)

def get_profiles(a):
    ns = a[CAM_ROW_START:CAM_ROW_END, CAM_NOSHIFT_C1:CAM_NOSHIFT_C2].mean(1)
    sh = a[CAM_ROW_START:CAM_ROW_END, CAM_SHIFT_C1:CAM_SHIFT_C2].mean(1)
    return ns, sh

def phase_of(frame):
    """SLM phase at this frame = find_phi(Shift) - find_phi(NoShift)."""
    ns, sh = get_profiles(np.asarray(frame, float))
    return find_phi(sh) - find_phi(ns)

def fringe_shift_px(frame0, frameG):
    """Backup metric: how far the Shift-ROI fringes moved (works at low contrast)."""
    _, s0 = get_profiles(np.asarray(frame0, float))
    _, sg = get_profiles(np.asarray(frameG, float))
    s0 = s0 - s0.mean(); sg = sg - sg.mean(); n = min(len(s0), len(sg))
    cc = np.correlate(s0[:n], sg[:n], "full")
    return int(cc.argmax() - (n - 1))


def square(gray):
    img = np.zeros((SLM_HEIGHT, SLM_WIDTH), np.uint8)
    img[RECT_ROW_START:RECT_ROW_END, RECT_COL_START:RECT_COL_END] = gray
    return img

def init_slm():
    sys.path.append(HOLOEYE_SDK_PATH)
    import HEDS
    from hedslib.heds_types import HEDSERR_NoError
    assert HEDS.SDK.Init(4, 0) == HEDSERR_NoError, "SDK init failed"
    slm = HEDS.SLM.Init("", True, 0.0); slm.setWavelength(LASER_WAVELENGTH)
    print(f"  SLM: {slm.width_px()}x{slm.height_px()} px")
    return HEDS, HEDSERR_NoError, slm

def show(slm, gray, NE):
    Image.fromarray(square(gray)).save(TEMP_BMP)
    err, h = (slm.loadPhaseDataFromFile(TEMP_BMP) if USE_PHASE_DATA
              else slm.loadImageDataFromFile(TEMP_BMP))
    if err == NE: h.show()

def init_cam():
    import PySpin
    system = PySpin.System.GetInstance(); cams = system.GetCameras()
    cam = cams.GetByIndex(0); cam.Init(); nm = cam.GetNodeMap()
    ea = PySpin.CEnumerationPtr(nm.GetNode("ExposureAuto"))
    ea.SetIntValue(ea.GetEntryByName("Off").GetValue())
    PySpin.CFloatPtr(nm.GetNode("ExposureTime")).SetValue(EXPOSURE_US)
    ac = PySpin.CEnumerationPtr(nm.GetNode("AcquisitionMode"))
    ac.SetIntValue(ac.GetEntryByName("Continuous").GetValue())
    cam.BeginAcquisition()
    for _ in range(WARMUP_FRAMES): cam.GetNextImage().Release()
    return PySpin, system, cams, cam

def grab(cam, settle):
    time.sleep(settle)
    for _ in range(FLUSH_FRAMES): cam.GetNextImage().Release()
    raw = cam.GetNextImage(); f = raw.GetNDArray().astype(np.uint8); raw.Release()
    return f

def cleanup(HEDS, PySpin, system, cams, cam):
    try:
        cam.EndAcquisition(); cam.DeInit(); del cam; cams.Clear(); system.ReleaseInstance()
    except Exception: pass
    try: HEDS.SDK.Close()
    except Exception: pass
    if os.path.exists(TEMP_BMP): os.remove(TEMP_BMP)


# ---- modes ----------------------------------------------------------------- #
def mode_live():
    os.makedirs(DIR, exist_ok=True)
    HEDS, NE, slm = init_slm()
    PySpin, system, cams, cam = init_cam()
    print("LIVE depth meter — rotate the Pluto polarizer to maximize. Ctrl-C to stop.\n")
    try:
        while True:
            show(slm, 0, NE);   f0 = grab(cam, min(SETTLE_TIME, 0.6))
            show(slm, 255, NE); f2 = grab(cam, min(SETTLE_TIME, 0.6))
            try:
                depth = abs(phase_of(f2) - phase_of(f0)) / np.pi
            except Exception:
                depth = float("nan")
            px = fringe_shift_px(f0, f2)
            frac = 0 if np.isnan(depth) else min(depth / TARGET_PI, 1.0)
            bar = "#" * int(frac * 40)
            print(f"\r depth = {depth:5.3f} pi  ({frac*100:4.0f}% of 2pi)  "
                  f"shift={px:+3d}px  |{bar:<40}|", end="", flush=True)
    except KeyboardInterrupt:
        print("\n stopped.")
    finally:
        cleanup(HEDS, PySpin, system, cams, cam)

def mode_sweep():
    os.makedirs(DIR, exist_ok=True)
    HEDS, NE, slm = init_slm()
    PySpin, system, cams, cam = init_cam()
    frames = {}
    try:
        for g in SWEEP_GRAYS:
            show(slm, g, NE); frames[g] = grab(cam, SETTLE_TIME)
            Image.fromarray(frames[g]).save(os.path.join(DIR, f"Capture_gray_{g:03d}.bmp"))
            print(f"  gray {g:3d} captured")
    finally:
        cleanup(HEDS, PySpin, system, cams, cam)
    report(frames)

def mode_analyze():
    frames = {}
    for g in SWEEP_GRAYS:
        p = os.path.join(DIR, f"Capture_gray_{g:03d}.bmp")
        if os.path.exists(p):
            frames[g] = np.asarray(Image.open(p).convert("L"), float)
    if len(frames) < 2:
        print("  need >=2 Capture_gray_*.bmp in", DIR); return
    report(frames)

def report(frames: dict):
    grays = sorted(frames)
    raw = []                                  # per-frame phase in RADIANS
    for g in grays:
        try: raw.append(phase_of(frames[g]))
        except Exception: raw.append(np.nan)
    raw = np.array(raw)
    valid = ~np.isnan(raw)
    # UNWRAP across the sweep (Jonas's step) — stitches the pi-ambiguity jumps
    # into a continuous curve. Needs dense sampling to work (see SWEEP_GRAYS).
    phis = np.full(raw.shape, np.nan)
    phis[valid] = np.unwrap(raw[valid], period=np.pi) / np.pi
    phis = phis - np.nanmin(phis)            # start at 0 for readability
    depth = np.nanmax(phis) - np.nanmin(phis)
    print("\n--- modulation depth ---")
    for g, p in zip(grays, phis):
        print(f"  gray {g:3d}: {p:+.3f} pi")
    print(f"  DEPTH (max-min) = {depth:.3f} pi   (need >= {TARGET_PI:.0f} pi to calibrate)")
    print("  -> " + ("enough range to calibrate ✓" if depth >= TARGET_PI
                     else "too small — increase it (polarization / config) before calibrating"))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(grays, phis, "o-")
        plt.axhline(TARGET_PI, color="r", ls="--", label=f"{TARGET_PI:.0f}π target")
        plt.xlabel("gray value"); plt.ylabel("phase [π]")
        plt.title(f"Phase vs gray — modulation depth = {depth:.3f} π")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        out = os.path.join(DIR, "modulation_depth.png"); plt.savefig(out, dpi=140)
        print("  saved", out)
    except Exception as e:
        print("  (plot skipped:", e, ")")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    mode = ("live" if a.live else "sweep" if a.sweep else "analyze" if a.analyze else MODE)
    {"live": mode_live, "sweep": mode_sweep, "analyze": mode_analyze}[mode]()


if __name__ == "__main__":
    main()
