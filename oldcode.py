import os
import sys
import time
import numpy as np
import cv2


def center_pattern_sml(pattern, width=1920, height=1200):

    n = pattern.shape[0]

    M = np.zeros((height, width), dtype=pattern.dtype)

    # centreren
    start_y = (height - n) // 2       # verticaal (rijen)
    end_y   = start_y + n

    start_x = (width - n) // 2        # horizontaal (kolommen)
    end_x   = start_x + n

    M[start_y:end_y, start_x:end_x] = pattern

    return M

sys.path.append(r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.0.0\examples")

try:
    import PySpin
    USE_CAMERA = True
    print("PySpin found.")
except ImportError:
    print("PySpin not found; running without camera.")
    USE_CAMERA = False


import HEDS
from hedslib.heds_types import *

OUTPUT_DIR = r"C:\Users\mu00129\Desktop\slmnew10\capturescameraConstant2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SLM_WIDTH = 1024
SLM_HEIGHT = 768
NUM_GRAY_LEVELS = 256

# -------------------------
# Initialize SDK and SLM
# -------------------------

# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window with default scale (1:1) for the selected SLM:
slm = HEDS.SLM.Init()
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Initialize Camera
# -------------------------
system = cams = camera = None
if USE_CAMERA:
    system = PySpin.System.GetInstance()
    cams = system.GetCameras()
    if cams.GetSize() > 0:
        camera = cams.GetByIndex(0)
        camera.Init()
        nodemap = camera.GetNodeMap()
        acq = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        ac_cont = acq.GetEntryByName("Continuous")
        acq.SetIntValue(ac_cont.GetValue())
        camera.BeginAcquisition()
        print("Camera initialized.")
    else:
        print("No camera found.")
        USE_CAMERA = False


    for gray_val in range(NUM_GRAY_LEVELS):

        img_array = np.zeros((SLM_HEIGHT, SLM_WIDTH), dtype=np.uint8)
        img_array[:, :SLM_WIDTH // 2] = 0
        img_array[:, SLM_WIDTH // 2:] = gray_val

        err, dataHandle = slm.loadImageData(img_array)
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
        err = dataHandle.show()
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

        slm_filename = f"SLM_gray_{gray_val:03d}.bmp"

        time.sleep(0.1)

        try:
            img = camera.GetNextImage()
            if not img.IsIncomplete():
                frame = img.GetNDArray()
                capture_filename = f"Capture_gray_{gray_val:03d}.bmp"
                cv2.imwrite(os.path.join(OUTPUT_DIR, capture_filename), frame)
                print(f"[{gray_val:03d}/255] Captured → {capture_filename}")
            img.Release()
        except Exception as e:
            print(f"Capture failed at gray={gray_val}: {e}")

    for gray_val in range(5):
        coarse = np.random.randint(0, 256, (1920, 1200), dtype=np.uint8)
        active = cv2.resize(coarse, (1920, 1200), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        pattern = center_pattern_sml(active)

        err, dataHandle = slm.loadImageData(pattern)
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
        err = dataHandle.show()
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

        slm_filename = f"SLM_PIC{gray_val}.bmp"
        slm_pattern_filename = f"SLM_Pattern{gray_val}.bmp"

        time.sleep(0.1)

        try:
            img = camera.GetNextImage()
            if not img.IsIncomplete():
                frame = img.GetNDArray()
                capture_filename = f"Capture_gray_{gray_val:03d}.bmp"
                cv2.imwrite(os.path.join(OUTPUT_DIR, capture_filename), frame)
                cv2.imwrite(os.path.join(OUTPUT_DIR, slm_pattern_filename), pattern)
                print(f"[{gray_val:03d}/255] Captured → {capture_filename}")
            img.Release()
        except Exception as e:
            print(f"Capture failed at gray={(gray_val+255)}: {e}")


# -------------------------
# Cleanup
# -------------------------
if USE_CAMERA and camera:
    try:
        camera.EndAcquisition()
        camera.DeInit()
    except:
        pass
    del camera
if USE_CAMERA and cams:
    cams.Clear()
    del cams
if USE_CAMERA and system:
    system.ReleaseInstance()

slm.close()
HEDS.SDK.Close()



