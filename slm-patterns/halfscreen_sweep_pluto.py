import sys
import time
import os
import numpy as np
from PIL import Image


OUTPUT_DIR       = r"C:\Users\mu00129\Desktop\slmnew10\capturesCameraHalfScreen"
HOLOEYE_SDK_PATH = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.0.0\examples"

SETTLE_TIME    = 0.15   
WARMUP_FRAMES  = 10     
EXPOSURE_US    = 85.0


os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Camera ---
try:
    import PySpin
    USE_CAMERA = True
    print("PySpin found.")
except ImportError:
    print("PySpin not found.")
    USE_CAMERA = False

system = None
cams   = None
camera = None

if USE_CAMERA:
    system = PySpin.System.GetInstance()
    cams   = system.GetCameras()
    if cams.GetSize() > 0:
        camera = cams.GetByIndex(0)
        camera.Init()
        nodemap = camera.GetNodeMap()

        exp_auto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
        exp_auto.SetIntValue(exp_auto.GetEntryByName("Off").GetValue())
        exp_time = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
        exp_time.SetValue(EXPOSURE_US)

        acq = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        acq.SetIntValue(acq.GetEntryByName("Continuous").GetValue())
        camera.BeginAcquisition()

        print(f"Discarding {WARMUP_FRAMES} warmup frames...")
        for _ in range(WARMUP_FRAMES):
            frm = camera.GetNextImage()
            frm.Release()
        print("Camera ready.")
    else:
        print("No camera detected.")
        USE_CAMERA = False

sys.path.append(HOLOEYE_SDK_PATH)
import HEDS
from hedslib.heds_types import *

err = HEDS.SDK.Init(4, 0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# SDK 4.0 correct init — matches imagedata_uint8.py example exactly
slm = HEDS.SLM.Init("", True, 0.0)
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

slm_width  = slm.width_px()
slm_height = slm.height_px()
print(f"SLM size: {slm_width} x {slm_height}")



def send_to_slm(gray_val):
    data = HEDS.SLMDataField(2, 1, HEDSDTFMT_INT_U8, HEDSSHF_PresentFitScreen)

    err = data.setPixel(0, 0, 0)          # left half = gray 0
    assert err == HEDSERR_NoError, f"setPixel failed: {err}"
    err = data.setPixel(1, 0, int(gray_val))   # right half = sweep
    assert err == HEDSERR_NoError, f"setPixel failed: {err}"

    err = slm.showImageData(data)
    if err != HEDSERR_NoError:
        print(f"  showImageData failed at gray {gray_val}: {err}")
        return False
    return True


def capture_image(gray_value):
    if not USE_CAMERA:
        return True
    try:
        raw = camera.GetNextImage()
        if raw.IsIncomplete():
            print(f"  WARNING: incomplete frame at gray {gray_value}")
            raw.Release()
            return False
        frame    = raw.GetNDArray().astype(np.uint8)
        filename = f"Capture_gray_{gray_value:03d}.bmp"
        Image.fromarray(frame).save(os.path.join(OUTPUT_DIR, filename))
        raw.Release()
        return True
    except Exception as e:
        print(f"  Capture failed at gray {gray_value}: {e}")
        return False


def run_sweep():
    print(f"\nStarting sweep: gray 0 to 255")
    print(f"Output: {OUTPUT_DIR}\n")

    failed = []

    for gray_val in range(256):
        ok = send_to_slm(gray_val)
        if not ok:
            failed.append(gray_val)
            continue

        time.sleep(SETTLE_TIME)

        ok = capture_image(gray_val)
        if not ok:
            failed.append(gray_val)

        if gray_val % 32 == 0 or gray_val == 255:
            print(f"  [{gray_val:3d}/255]  {'OK' if ok else 'FAILED'}")

    print(f"\nSweep done. Captured: {256 - len(failed)} / 256")
    if failed:
        print(f"Failed: {failed}")


def cleanup():
    print("\nCleaning up...")
    try:
        HEDS.SDK.Close()
    except Exception as e:
        print(f"  SDK close: {e}")

    if USE_CAMERA and camera is not None:
        try:
            camera.EndAcquisition()
            camera.DeInit()
        except Exception as e:
            print(f"  Camera: {e}")
    if USE_CAMERA and cams is not None:
        cams.Clear()
    if USE_CAMERA and system is not None:
        system.ReleaseInstance()
    print("Done.")


try:
    run_sweep()
finally:
    cleanup()
