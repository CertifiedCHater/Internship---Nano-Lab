import os
import sys
import time
import numpy as np
from PIL import Image


OUTPUT_DIR = r"C:\Users\mu00129\Desktop\slmnew10\Measurements2\dual_slm"
HOLOEYE_SDK_PATH = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.0.0\examples"
LASER_WAVELENGTH = 633.0


ERIS_SELECT  = "ERIS"      
PLUTO_SELECT = "PLUTO"     

ERIS_SQUARE   = 400
ERIS_USE_PHASE_DATA = False     

PLUTO_GRAY    = 255
PLUTO_USE_PHASE_DATA = True      

NUM_GRAY_LEVELS = 256
EXPOSURE_US   = 85.0
WARMUP_FRAMES = 20
SETTLE_TIME   = 1.0
FLUSH_FRAMES  = 3

CALIB_PREFIX = "Capture_gray_"
CALIB_SUFFIX = ".bmp"
TMP_ERIS  = os.path.join(OUTPUT_DIR, "_tmp_eris.bmp")
TMP_PLUTO = os.path.join(OUTPUT_DIR, "_tmp_pluto.bmp")



def full_screen(width, height, gray):
    return np.full((height, width), gray, dtype=np.uint8)


def centered_square(width, height, size, gray):
    img = np.zeros((height, width), dtype=np.uint8)
    y0 = (height - size) // 2
    x0 = (width - size) // 2
    img[y0:y0 + size, x0:x0 + size] = gray
    return img


def init_slm(HEDS, HEDSERR_NoError, select, label):
    slm = HEDS.SLM.Init(select, True, 0.0)
    assert slm.errorCode() == HEDSERR_NoError, \
        f"{label} init failed ({slm.errorCode()}); check {label}_SELECT."
    slm.setWavelength(LASER_WAVELENGTH)
    print(f"  {label}: {slm.width_px()} x {slm.height_px()} px")
    return slm


def show_pattern(slm, pattern, use_phase_data, tmp_path, HEDSERR_NoError):
    Image.fromarray(pattern).save(tmp_path)
    if use_phase_data:
        err, handle = slm.loadPhaseDataFromFile(tmp_path)
    else:
        err, handle = slm.loadImageDataFromFile(tmp_path)
    if err != HEDSERR_NoError:
        print(f"  load failed: {err}")
        return False
    handle.show()
    return True


def init_camera():
    import PySpin
    system = PySpin.System.GetInstance()
    cams = system.GetCameras()
    assert cams.GetSize() > 0, "No camera found"
    cam = cams.GetByIndex(0); cam.Init()
    nm = cam.GetNodeMap()
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

    pluto = init_slm(HEDS, HEDSERR_NoError, PLUTO_SELECT, "PLUTO")
    eris  = init_slm(HEDS, HEDSERR_NoError, ERIS_SELECT,  "ERIS")

    pw, ph = pluto.width_px(), pluto.height_px()
    show_pattern(pluto, full_screen(pw, ph, PLUTO_GRAY),
                 PLUTO_USE_PHASE_DATA, TMP_PLUTO, HEDSERR_NoError)
    print(f"  PLUTO held at full-screen gray {PLUTO_GRAY}")

    PySpin, system, cams, cam = init_camera()
    print("  camera ready")

    ew, eh = eris.width_px(), eris.height_px()
    print(f"  sweeping ERIS {ERIS_SQUARE}x{ERIS_SQUARE} square 0..255")
    for gray in range(NUM_GRAY_LEVELS):
        show_pattern(eris, centered_square(ew, eh, ERIS_SQUARE, gray),
                     ERIS_USE_PHASE_DATA, TMP_ERIS, HEDSERR_NoError)
        time.sleep(SETTLE_TIME)
        frame = grab_frame(cam)
        if frame is not None:
            Image.fromarray(frame).save(
                os.path.join(OUTPUT_DIR, f"{CALIB_PREFIX}{gray:03d}{CALIB_SUFFIX}"))
        if gray % 32 == 0 or gray == 255:
            print(f"  [{gray:3d}/255]")

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
