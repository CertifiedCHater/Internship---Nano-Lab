from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np
from PIL import Image


OUTPUT_DIR = r"C:\Users\mu00129\Desktop\slmnew10\Measurements2\overlap"
SLM_WIDTH, SLM_HEIGHT = 1920, 1200      
SQUARE   = 300                          
STEP     = 150                          
SQUARE_GRAY = 255                       

HOLOEYE_SDK_PATH = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.0.0\examples"
LASER_WAVELENGTH = 633.0
SETTLE_TIME, FLUSH_FRAMES, WARMUP_FRAMES, EXPOSURE_US = 1.0, 5, 20, 85.0
USE_PHASE_DATA = False                  
TEMP_BMP = os.path.join(OUTPUT_DIR, "_tmp.bmp")



def _init_slm():
    sys.path.append(HOLOEYE_SDK_PATH)
    import HEDS
    from hedslib.heds_types import HEDSERR_NoError
    assert HEDS.SDK.Init(4, 0) == HEDSERR_NoError, "SDK Init failed"
    slm = HEDS.SLM.Init("", True, 0.0)
    slm.setWavelength(LASER_WAVELENGTH)
    print(f"  SLM: {slm.width_px()}x{slm.height_px()} px")
    return HEDS, HEDSERR_NoError, slm

def _show(slm, img, HEDSERR_NoError):
    Image.fromarray(img).save(TEMP_BMP)
    err, h = (slm.loadPhaseDataFromFile(TEMP_BMP) if USE_PHASE_DATA
              else slm.loadImageDataFromFile(TEMP_BMP))
    if err == HEDSERR_NoError:
        h.show()

def _init_cam():
    import PySpin
    system = PySpin.System.GetInstance(); cams = system.GetCameras()
    cam = cams.GetByIndex(0); cam.Init(); nm = cam.GetNodeMap()
    ea = PySpin.CEnumerationPtr(nm.GetNode("ExposureAuto"))
    ea.SetIntValue(ea.GetEntryByName("Off").GetValue())
    PySpin.CFloatPtr(nm.GetNode("ExposureTime")).SetValue(EXPOSURE_US)
    ac = PySpin.CEnumerationPtr(nm.GetNode("AcquisitionMode"))
    ac.SetIntValue(ac.GetEntryByName("Continuous").GetValue())
    cam.BeginAcquisition()
    for _ in range(WARMUP_FRAMES):
        cam.GetNextImage().Release()
    return PySpin, system, cams, cam

def _grab(cam):
    for _ in range(FLUSH_FRAMES):
        cam.GetNextImage().Release()
    raw = cam.GetNextImage()
    f = raw.GetNDArray().astype(np.uint8)
    raw.Release()
    return f

def square_at(top, left, gray=SQUARE_GRAY):
    img = np.zeros((SLM_HEIGHT, SLM_WIDTH), np.uint8)
    img[top:top + SQUARE, left:left + SQUARE] = gray
    return img


def grid_positions():
    tops  = list(range(0, SLM_HEIGHT - SQUARE + 1, STEP))
    lefts = list(range(0, SLM_WIDTH  - SQUARE + 1, STEP))
    return tops, lefts

def cap_pluto_ref(out=OUTPUT_DIR):
    os.makedirs(out, exist_ok=True)
    HEDS, NE, slm = _init_slm()
    _show(slm, np.zeros((SLM_HEIGHT, SLM_WIDTH), np.uint8), NE)   # ERIS black
    PySpin, system, cams, cam = _init_cam()
    input("  Turn the PLUTO square OFF (all black) in the GUI, then press Enter...")
    Image.fromarray(_grab(cam)).save(os.path.join(out, "pluto_off.bmp"))
    input("  Turn the PLUTO square ON in the GUI, then press Enter...")
    Image.fromarray(_grab(cam)).save(os.path.join(out, "pluto_on.bmp"))
    _cleanup(HEDS, PySpin, system, cams, cam)
    print("  saved pluto_off.bmp / pluto_on.bmp")

def cap_sweep(out=OUTPUT_DIR):
    os.makedirs(out, exist_ok=True)
    HEDS, NE, slm = _init_slm()
    PySpin, system, cams, cam = _init_cam()
    # background: ERIS fully black (PLUTO stays whatever it is)
    _show(slm, np.zeros((SLM_HEIGHT, SLM_WIDTH), np.uint8), NE)
    time.sleep(SETTLE_TIME)
    Image.fromarray(_grab(cam)).save(os.path.join(out, "eris_bg.bmp"))
    tops, lefts = grid_positions()
    print(f"  sweeping {len(tops)}x{len(lefts)} = {len(tops)*len(lefts)} positions")
    for iy, t in enumerate(tops):
        for ix, l in enumerate(lefts):
            _show(slm, square_at(t, l), NE)
            time.sleep(SETTLE_TIME)
            f = _grab(cam)
            Image.fromarray(f).save(os.path.join(out, f"eris_{iy:02d}_{ix:02d}.bmp"))
        print(f"    row {iy+1}/{len(tops)} done")
    _cleanup(HEDS, PySpin, system, cams, cam)
    print("  sweep complete")

def _cleanup(HEDS, PySpin, system, cams, cam):
    try:
        cam.EndAcquisition(); cam.DeInit(); del cam
        cams.Clear(); system.ReleaseInstance()
    except Exception:
        pass
    try:
        HEDS.SDK.Close()
    except Exception:
        pass
    if os.path.exists(TEMP_BMP):
        os.remove(TEMP_BMP)


def _load(p):
    return np.asarray(Image.open(p).convert("L"), float)

def _footprint(diff, frac=0.5):
    """Return (cy, cx, mask) of the strongest changed blob in a difference image."""
    from scipy.ndimage import gaussian_filter
    d = gaussian_filter(np.abs(diff), 8)
    mask = d > d.max() * frac
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return ys.mean(), xs.mean(), mask

def analyze(out=OUTPUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_on, p_off = os.path.join(out, "pluto_on.bmp"), os.path.join(out, "pluto_off.bmp")
    if not (os.path.exists(p_on) and os.path.exists(p_off)):
        print("  missing pluto_on/off.bmp — run --pluto-ref first."); return
    fp_p = _footprint(_load(p_on) - _load(p_off))
    if fp_p is None:
        print("  could not find PLUTO footprint (its modulation may be too weak)."); return
    pcy, pcx, pmask = fp_p

    bg = _load(os.path.join(out, "eris_bg.bmp"))
    best = None
    for f in sorted(glob.glob(os.path.join(out, "eris_[0-9]*_[0-9]*.bmp"))):
        fp = _footprint(_load(f) - bg)
        if fp is None:
            continue
        cy, cx, _ = fp
        dist = np.hypot(cy - pcy, cx - pcx)
        if best is None or dist < best[0]:
            best = (dist, f, cy, cx)

    if best is None:
        print("  no ERIS footprints detected — ERIS square may be off the camera."); return
    dist, fpath, ecy, ecx = best
    name = os.path.basename(fpath)
    print(f"  PLUTO square center on camera : ({pcx:.0f}, {pcy:.0f})")
    print(f"  best-matching ERIS frame      : {name}")
    print(f"  its center on camera          : ({ecx:.0f}, {ecy:.0f})")
    print(f"  centroid distance             : {dist:.0f} px  (smaller = better aligned)")

    eris_best = _load(fpath); _, _, emask = _footprint(eris_best - bg)
    rgb = np.stack([_load(p_on)] * 3, -1); rgb = rgb / rgb.max()
    rgb[pmask] = [1, 0, 0]          # PLUTO footprint in red
    rgb[emask] = [0, 1, 0]          # ERIS footprint in green
    plt.figure(figsize=(7, 7)); plt.imshow(rgb)
    plt.title(f"PLUTO (red) vs best ERIS (green)\n{name}, centroid gap {dist:.0f}px")
    plt.axis("off"); plt.tight_layout()
    o = os.path.join(out, "overlap_result.png"); plt.savefig(o, dpi=130)
    print(f"  saved {o}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pluto-ref", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--dir", default=OUTPUT_DIR)
    a = ap.parse_args()
    if a.pluto_ref: cap_pluto_ref(a.dir)
    elif a.sweep:   cap_sweep(a.dir)
    elif a.analyze: analyze(a.dir)
    else:           print("pick one: --pluto-ref | --sweep | --analyze")


if __name__ == "__main__":
    main()



