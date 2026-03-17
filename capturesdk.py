# import os
# import sys
# import time
# import numpy as np
# import cv2

# sys.path.append(r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.1.0\api")
# import slmdisplaysdk

# try:
#     import PySpin
#     USE_CAMERA = True
# except ImportError:
#     print("PySpin not found; running in display-only mode.")
#     USE_CAMERA = False


# OUTPUT_DIR = r"C:\SLM_capture"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# NUM_IMAGES = 255

# # Initialize SLM
# slm = slmdisplaysdk.SLMDisplay()
# error = slm.open()

# if error != slmdisplaysdk.ErrorCode.NoError:
#     print("Could not open SLM")
#     sys.exit()

# width = slm.width_px
# height = slm.height_px

# print("SLM resolution:", width, height)

# # Initialize camera
# camera = None
# system = cams = None

# if USE_CAMERA:
#     system = PySpin.System.GetInstance()
#     cams = system.GetCameras()

#     if cams.GetSize() > 0:
#         camera = cams.GetByIndex(0)
#         camera.Init()

#         nodemap = camera.GetNodeMap()

#         acq = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
#         ac_cont = acq.GetEntryByName("Continuous")
#         acq.SetIntValue(ac_cont.GetValue())

#         camera.BeginAcquisition()
#         print("Camera initialized")

#     else:
#         print("No camera found")
#         USE_CAMERA = False


# for i in range(NUM_IMAGES):

#     gray_val = i

#     img_array = np.zeros((height, width), dtype=np.uint8)

#     img_array[:, :width//2] = 255
#     img_array[:, width//2:] = gray_val

#     slm.showData(img_array)

#     time.sleep(0.05)

#     bmp_arr = None

#     if camera:
#         img = camera.GetNextImage()

#         if not img.IsIncomplete():
#             conv = img.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
#             bmp_arr = conv.GetNDArray()

#         img.Release()

#     timestamp = time.strftime("%Y%m%d_%H%M%S")

#     pattern_path = os.path.join(OUTPUT_DIR, f"{timestamp}_pattern_{i:03d}.bmp")
#     cv2.imwrite(pattern_path, img_array)

#     if bmp_arr is not None:
#         capture_path = os.path.join(OUTPUT_DIR, f"{timestamp}_capture_{i:03d}.bmp")
#         cv2.imwrite(capture_path, bmp_arr)

#     print(f"{i+1}/{NUM_IMAGES} gray={gray_val}")


# slm.close()

# if camera:
#     camera.EndAcquisition()
#     camera.DeInit()

# if cams:
#     cams.Clear()

# if system:

#     system.ReleaseInstance()


import os
import sys
import time
import numpy as np
import cv2

def center_pattern_slm(pattern, width=1024, height=768):
#   Place a n×n pattern centered in an SLM of size height×width.

#   pattern: 2D array (n×n)
#   width: total width of the SLM (e.g. 1024)
#   height: total height of the SLM (e.g. 768)

    n = pattern.shape[0]

    # matrix sent to the SLM, initialized with zeros (black)
    M = np.zeros((height, width), dtype=pattern.dtype)

    # centre
    start_y = (height - n) // 2       # vertical (rows)
    end_y   = start_y + n

    start_x = (width - n) // 2        # horizontal (columns)
    end_x   = start_x + n

    M[start_y:end_y, start_x:end_x] = pattern

    return M

sys.path.append(r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.1.0\api")

try:
    import PySpin
    USE_CAMERA = True
    print("PySpin found.")
except ImportError:
    print("PySpin not found; running without camera.")
    USE_CAMERA = False


import HEDS
from hedslib.heds_types import *

OUTPUT_DIR = r"D:\Saxion\Junior\Internship - Nano Lab\Work\Images"  # "Personal path, on laptop for testing"
# OUTPUT_DIR = r"C:\Users\mu00129\Desktop\slm_patterns"         Lab path, will be fixed into the lab computer after finished testing
os.makedirs(OUTPUT_DIR, exist_ok=True)

SLM_WIDTH = 1024
SLM_HEIGHT = 768
NUM_PATTERNS = 256

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


    for pattern_idx in range(NUM_PATTERNS):

        img_array = np.zeros((SLM_HEIGHT, SLM_WIDTH), dtype=np.uint8)
        img_array[:, :SLM_WIDTH // 4] = 0
        img_array[:, SLM_WIDTH // 4:3*SLM_WIDTH // 4] = pattern_idx
        img_array[:, 3*SLM_WIDTH // 4:] = 0

        err, dataHandle = slm.loadImageData(img_array)
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

        err = dataHandle.show()
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

        slm_filename = f"SLM_pattern_{pattern_idx:03d}.bmp"

        time.sleep(0.1)

        try:
            img = camera.GetNextImage()
            if not img.IsIncomplete():
                frame = img.GetNDArray()
                capture_filename = f"Capture_pattern_{pattern_idx:03d}.bmp"
                cv2.imwrite(os.path.join(OUTPUT_DIR, capture_filename), frame)
                print(f"[{pattern_idx:03d}/255] Captured → {capture_filename}")
            img.Release()
        except Exception as e:
            print(f"Capture failed at pattern={pattern_idx}: {e}")

    for pattern_idx in range(5):
        coarse = np.random.randint(0, 256, (1024, 768), dtype=np.uint8)
        active = cv2.resize(coarse, (1024, 768), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        pattern = center_pattern_slm(active)

        err, dataHandle = slm.loadImageData(pattern)
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
        err = dataHandle.show()
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

        slm_filename = f"SLM_PIC{pattern_idx}.bmp"
        slm_pattern_filename = f"SLM_Pattern{pattern_idx}.bmp"

        time.sleep(0.1)

        try:
            img = camera.GetNextImage()
            if not img.IsIncomplete():
                frame = img.GetNDArray()
                capture_filename = f"Capture_pattern_{pattern_idx:03d}.bmp"
                cv2.imwrite(os.path.join(OUTPUT_DIR, capture_filename), frame)
                cv2.imwrite(os.path.join(OUTPUT_DIR, slm_pattern_filename), pattern)
                print(f"[{pattern_idx:03d}/255] Captured → {capture_filename}")
            img.Release()
        except Exception as e:
            print(f"Capture failed at pattern={(pattern_idx + 255)}: {e}")


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

