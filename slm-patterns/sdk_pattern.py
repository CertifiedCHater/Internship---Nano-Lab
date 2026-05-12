import sys
import time
import os
import numpy as np
from PIL import Image


OUTPUT_DIR       = r"C:\Users\mu00129\Desktop\slmnew10\capturesCameraSquareNew3"
HOLOEYE_SDK_PATH = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.0.0\examples"


RECT_X      = 1000
RECT_Y      = 600
RECT_WIDTH  = 300
RECT_HEIGHT = 300


SETTLE_TIME = 0.15  

EXPOSURE_US = 85.0

WARMUP_FRAMES = 20



os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Camera setup ---
try:
    import PySpin
    USE_CAMERA = True
    print("PySpin found.")
except ImportError:
    print("PySpin not found — running without camera.")
    USE_CAMERA = False

system = cams = camera = None

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
            img = camera.GetNextImage()
            img.Release()

        print("Camera ready.")
    else:
        print("No camera detected.")
        USE_CAMERA = False

sys.path.append(HOLOEYE_SDK_PATH)
import HEDS
from hedslib.heds_types import HEDSERR_NoError

err = HEDS.SDK.Init(4, 0)
assert err == HEDSERR_NoError, f"SDK init failed: {err}"


devices = HEDS.SLM.GetDisplays()
print(f"\nAvailable SLM devices: {len(devices)}")
for i, d in enumerate(devices):
    print(f"  [{i}] {d}")

slm = HEDS.SLM.Init(0)
assert slm.errorCode() == HEDSERR_NoError, f"SLM init failed: {slm.errorCode()}"

slm_width  = slm.width_px()
slm_height = slm.height_px()
print(f"\nSLM size: {slm_width} x {slm_height}")
print(f"Square: x={RECT_X}, y={RECT_Y}, w={RECT_WIDTH}, h={RECT_HEIGHT}\n")



def make_slm_image(gray_value):
    img = np.zeros((slm_height, slm_width), dtype=np.uint8)
    img[RECT_Y : RECT_Y + RECT_HEIGHT,
        RECT_X : RECT_X + RECT_WIDTH] = gray_value
    return img


def send_to_slm(gray_value):
    img = make_slm_image(gray_value)
    print(f"Array check: background={img[0,0]}, "
      f"square center={img[RECT_Y + RECT_HEIGHT//2, RECT_X + RECT_WIDTH//2]}, "
      f"non-zero pixels={np.count_nonzero(img)}")
    err, handle = slm.loadImageData(img)
    if err != HEDSERR_NoError:
        print(f"  SLM load error {err} at gray {gray_value}")
        return False
    err = handle.show()
    if err != HEDSERR_NoError:
        print(f"  SLM show error {err} at gray {gray_value}")
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
        filepath = os.path.join(OUTPUT_DIR, filename)
        Image.fromarray(frame).save(filepath)
        raw.Release()
        return True

    except Exception as e:
        print(f"  Capture failed at gray {gray_value}: {e}")
        return False


def run_sweep():
    print(f"Starting sweep: gray 0 to 255")
    print(f"Output: {OUTPUT_DIR}\n")

    failed = []

    for gray_value in range(256):

        ok = send_to_slm(gray_value)
        if not ok:
            failed.append(gray_value)
            continue

        time.sleep(SETTLE_TIME)

        ok = capture_image(gray_value)
        if not ok:
            failed.append(gray_value)

        print(f"  gray {gray_value:03d}/255  {'OK' if ok else 'FAILED'}")

    print(f"\nSweep done. Captured: {256 - len(failed)}/256")
    if failed:
        print(f"Failed gray values: {failed}")



def cleanup():
    print("\nCleaning up...")
    try:
        slm.close()
        HEDS.SDK.Close()
    except Exception as e:
        print(f"  SLM cleanup: {e}")

    if USE_CAMERA and camera:
        try:
            camera.EndAcquisition()
            camera.DeInit()
            del camera
        except Exception as e:
            print(f"  Camera cleanup: {e}")

    if USE_CAMERA and cams:
        cams.Clear()
    if USE_CAMERA and system:
        system.ReleaseInstance()

    print("Done.")

try:
    run_sweep()
finally:
    cleanup()
