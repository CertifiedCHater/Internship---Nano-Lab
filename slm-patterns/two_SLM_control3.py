import os
import sys
import time
import numpy as np
from PIL import Image

OUTPUT_DIR = r"C:\Users\mu00129\Desktop\slmnew10\Measurements2\dual_split_sweep"
HOLOEYE_SDK_PATH = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.0.0\examples"
LASER_WAVELENGTH = 633.0

ERIS_SELECT  = "ERIS"
PLUTO_SELECT = "PLUTO"

ERIS_W,  ERIS_H  = 1920, 1200
PLUTO_W, PLUTO_H = 1920, 1080

ERIS_SIZE   = 400
ERIS_CENTER = (950, 600)      
ERIS_GRAY   = 255


PLUTO_SPLIT_COL = PLUTO_W // 2   # boundary column: left = fading, right = white
RIGHT_GRAY      = 255            

ERIS_USE_PHASE_DATA  = False
PLUTO_USE_PHASE_DATA = True

EXPOSURE_US   = 85.0
WARMUP_FRAMES = 20
SETTLE_TIME   = 1.0
FLUSH_FRAMES  = 3

CALIB_PREFIX = "Capture_gray_"
CALIB_SUFFIX = ".bmp"
TMP_ERIS  = os.path.join(OUTPUT_DIR, "_tmp_eris.bmp")
TMP_PLUTO = os.path.join(OUTPUT_DIR, "_tmp_pluto.bmp")


def eris_square():
    """Fixed ERIS square (gray 255 at ERIS_CENTER)."""
    img = np.zeros((ERIS_H, ERIS_W), dtype=np.uint8)
    cx, cy = ERIS_CENTER
    x0 = max(0, cx - ERIS_SIZE // 2); y0 = max(0, cy - ERIS_SIZE // 2)
    img[y0:min(ERIS_H, y0 + ERIS_SIZE), x0:min(ERIS_W, x0 + ERIS_SIZE)] = ERIS_GRAY
    return img


def pluto_split(left_gray):
    """PLUTO split: right half = RIGHT_GRAY (white), left half = left_gray."""
    img = np.zeros((PLUTO_H, PLUTO_W), dtype=np.uint8)
    img[:, PLUTO_SPLIT_COL:] = RIGHT_GRAY      # right half white
    img[:, :PLUTO_SPLIT_COL] = left_gray       # left half (fades)
    return img


def init_slm(HEDS, HEDSERR_NoError, select, label, cfg_w, cfg_h):
    slm = HEDS.SLM.Init(select, True, 0.0)
    assert slm.errorCode() == HEDSERR_NoError, \
        f"{label} init failed ({slm.errorCode()}); check {label}_SELECT."
    slm.setWavelength(LASER_WAVELENGTH)
    w, h = slm.width_px(), slm.height_px()
    print(f"  {label}: SDK reports {w} x {h} px  (your config: {cfg_w} x {cfg_h})")
    if (w, h) != (cfg_w, cfg_h):
        print(f"  ** {label}: config size != reported size — set {label}_W/{label}_H to {w}x{h}.")
    return slm


def show_pattern(slm, pattern, use_phase_data, tmp_path, HEDSERR_NoError):
    Image.fromarray(pattern).save(tmp_path)
    if use_phase_data:
        err, handle = slm.loadPhaseDataFromFile(tmp_path)
    else:
        err, handle = slm.loadImageDataFromFile(tmp_path)
    if err != HEDSERR_NoError:
        print(f"  load failed: {err}"); return False
    handle.show(); return True


def init_camera():
    import PySpin
    system = PySpin.System.GetInstance(); cams = system.GetCameras()
    assert cams.GetSize() > 0, "No camera found"
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


def grab_frame(cam):
    for _ in range(FLUSH_FRAMES):
        cam.GetNextImage().Release()
    raw = cam.GetNextImage()
    frame = raw.GetNDArray().astype(np.uint8) if not raw.IsIncomplete() else None
    raw.Release()
    return frame


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sys.path.append(HOLOEYE_SDK_PATH)
    import HEDS
    from hedslib.heds_types import HEDSERR_NoError
    assert HEDS.SDK.Init(4, 0) == HEDSERR_NoError, "SDK Init failed"

    pluto = init_slm(HEDS, HEDSERR_NoError, PLUTO_SELECT, "PLUTO", PLUTO_W, PLUTO_H)
    eris  = init_slm(HEDS, HEDSERR_NoError, ERIS_SELECT,  "ERIS",  ERIS_W,  ERIS_H)

    show_pattern(eris, eris_square(), ERIS_USE_PHASE_DATA, TMP_ERIS, HEDSERR_NoError)
    print(f"  ERIS: {ERIS_SIZE}x{ERIS_SIZE} square at {ERIS_CENTER}, gray {ERIS_GRAY}")

    PySpin, system, cams, cam = init_camera()
    print("  camera ready")

    print(f"  PLUTO split at col {PLUTO_SPLIT_COL}: right=white, left fades 255->0")
    for left_gray in range(255, -1, -1):        # white -> black over 256 steps
        show_pattern(pluto, pluto_split(left_gray), PLUTO_USE_PHASE_DATA, TMP_PLUTO, HEDSERR_NoError)
        time.sleep(SETTLE_TIME)
        frame = grab_frame(cam)
        if frame is not None:
            Image.fromarray(frame).save(
                os.path.join(OUTPUT_DIR, f"{CALIB_PREFIX}{left_gray:03d}{CALIB_SUFFIX}"))
        if left_gray % 32 == 0 or left_gray == 255:
            print(f"  left = {left_gray:3d}")

    try:
        cam.EndAcquisition(); cam.DeInit(); del cam
        cams.Clear(); system.ReleaseInstance()
    except Exception as e:
        print(f"  camera cleanup: {e}")
    try:
        HEDS.SDK.Close()
    except Exception as e:
        print(f"  SDK cleanup: {e}")
    for t in (TMP_ERIS, TMP_PLUTO):
        if os.path.exists(t):
            os.remove(t)
    print("  done ->", OUTPUT_DIR)


if __name__ == "__main__":
    run()
