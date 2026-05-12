import sys
import time
import os
import numpy as np
from PIL import Image



OUTPUT_DIR       = r"C:\Users\mu00129\Desktop\slmnew10\capturesCameraSquareNew3"
HOLOEYE_SDK_PATH = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.0.0\examples"

RECT_X      = 810    # (1920 - 300) // 2  = 810
RECT_Y      = 450    # (1200 - 300) // 2  = 450
RECT_WIDTH  = 300
RECT_HEIGHT = 300

# --- Timing ---
SETTLE_TIME   = 0.15   # seconds to wait after each SLM update
WARMUP_FRAMES = 10     # camera warmup frames to discard

# --- Camera ---
EXPOSURE_US = 85.0



os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Camera ---
try:
    import PySpin
    USE_CAMERA = True
    print("PySpin found.")
except ImportError:
    print("PySpin not found — running without camera.")
    USE_CAMERA = False

system = None
cams   = None
camera = None       # declared here so cleanup() always has it in scope

if USE_CAMERA:
    system = PySpin.System.GetInstance()
    cams   = system.GetCameras()

    if cams.GetSize() > 0:
        camera = cams.GetByIndex(0)
        camera.Init()
        nodemap = camera.GetNodeMap()

        # Disable auto exposure
        exp_auto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
        exp_auto.SetIntValue(exp_auto.GetEntryByName("Off").GetValue())

        # Set manual exposure time
        exp_time = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
        exp_time.SetValue(EXPOSURE_US)

        # Set continuous acquisition
        acq = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        acq.SetIntValue(acq.GetEntryByName("Continuous").GetValue())

        camera.BeginAcquisition()

        # Discard warmup frames (FLIR Grasshopper quirk)
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
from hedslib.heds_types import HEDSERR_NoError

err = HEDS.SDK.Init(4, 0)
assert err == HEDSERR_NoError, f"SDK init failed: {err}"

slm = HEDS.SLM.Init(0)     # index 0 = first SLM; change to 1 if it picks wrong one
assert slm.errorCode() == HEDSERR_NoError, f"SLM init failed: {slm.errorCode()}"

slm_width  = slm.width_px()
slm_height = slm.height_px()
print(f"SLM size: {slm_width} x {slm_height}")

assert RECT_X >= 0, \
    f"RECT_X={RECT_X} must be >= 0"
assert RECT_Y >= 0, \
    f"RECT_Y={RECT_Y} must be >= 0"
assert RECT_X + RECT_WIDTH <= slm_width, \
    (f"Rectangle too wide: RECT_X({RECT_X}) + RECT_WIDTH({RECT_WIDTH}) "
     f"= {RECT_X+RECT_WIDTH}, but SLM is only {slm_width}px wide")
assert RECT_Y + RECT_HEIGHT <= slm_height, \
    (f"Rectangle too tall: RECT_Y({RECT_Y}) + RECT_HEIGHT({RECT_HEIGHT}) "
     f"= {RECT_Y+RECT_HEIGHT}, but SLM is only {slm_height}px tall")

print(f"Square: x={RECT_X}, y={RECT_Y}, w={RECT_WIDTH}, h={RECT_HEIGHT}")
print(f"        right edge  = {RECT_X+RECT_WIDTH} / {slm_width}  ✓")
print(f"        bottom edge = {RECT_Y+RECT_HEIGHT} / {slm_height}  ✓")



def make_slm_image(gray_value):
    """
    Black background with a square at gray_value.
    Background = 0 (phase 0, static reference)
    Square     = gray_value (the phase being swept)
    """
    img = np.zeros((slm_height, slm_width), dtype=np.uint8)
    img[RECT_Y : RECT_Y + RECT_HEIGHT,
        RECT_X : RECT_X + RECT_WIDTH] = gray_value
    return img


def send_to_slm(gray_value):
    img = make_slm_image(gray_value)

    if gray_value == 1:
        bg      = img[0, 0]
        center  = img[RECT_Y + RECT_HEIGHT // 2, RECT_X + RECT_WIDTH // 2]
        nonzero = np.count_nonzero(img)
        expected = RECT_WIDTH * RECT_HEIGHT
        print(f"  Array check at gray 1:")
        print(f"    background pixel [0,0]   = {bg}       (expected 0)")
        print(f"    square center pixel      = {center}   (expected 1)")
        print(f"    non-zero pixels          = {nonzero}  (expected {expected})")

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

    for gray_value in range(256):

        # Send pattern to SLM (only once per gray value — not 60x per second)
        ok = send_to_slm(gray_value)
        if not ok:
            failed.append(gray_value)
            continue

        time.sleep(SETTLE_TIME)

        ok = capture_image(gray_value)
        if not ok:
            failed.append(gray_value)

        if gray_value % 32 == 0 or gray_value == 255:
            print(f"  [{gray_value:3d}/255]  {'OK' if ok else 'FAILED'}")

    print(f"\nSweep done.  Captured: {256 - len(failed)} / 256 images")
    if failed:
        print(f"Failed gray values: {failed}")



def cleanup():
    print("\nCleaning up...")

    try:
        slm.close()
        HEDS.SDK.Close()
    except Exception as e:
        print(f"  SLM cleanup: {e}")

    if USE_CAMERA and camera is not None:
        try:
            camera.EndAcquisition()
            camera.DeInit()
        except Exception as e:
            print(f"  Camera DeInit: {e}")

    if USE_CAMERA and cams is not None:
        try:
            cams.Clear()
        except Exception as e:
            print(f"  Cams clear: {e}")

    if USE_CAMERA and system is not None:
        try:
            system.ReleaseInstance()
        except Exception as e:
            print(f"  System release: {e}")

    print("Done.")

try:
    run_sweep()
finally:
    cleanup()
