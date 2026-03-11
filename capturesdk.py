import os
import sys
import time
import numpy as np
import cv2

sys.path.append(r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.1.0\api")
import slmdisplaysdk

try:
    import PySpin
    USE_CAMERA = True
except ImportError:
    print("PySpin not found; running in display-only mode.")
    USE_CAMERA = False


OUTPUT_DIR = r"C:\SLM_capture"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_IMAGES = 1000

# Initialize SLM
slm = slmdisplaysdk.SLMDisplay()
error = slm.open()

if error != slmdisplaysdk.ErrorCode.NoError:
    print("Could not open SLM")
    sys.exit()

width = slm.width_px
height = slm.height_px

print("SLM resolution:", width, height)

# Initialize camera
camera = None
system = cams = None

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
        print("Camera initialized")

    else:
        print("No camera found")
        USE_CAMERA = False


for i in range(NUM_IMAGES):

    gray_val = i

    img_array = np.zeros((height, width), dtype=np.uint8)

    img_array[:, :width//2] = 255
    img_array[:, width//2:] = gray_val

    slm.showData(img_array)

    time.sleep(0.05)

    bmp_arr = None

    if camera:
        img = camera.GetNextImage()

        if not img.IsIncomplete():
            conv = img.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
            bmp_arr = conv.GetNDArray()

        img.Release()

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    pattern_path = os.path.join(OUTPUT_DIR, f"{timestamp}_pattern_{i:03d}.bmp")
    cv2.imwrite(pattern_path, img_array)

    if bmp_arr is not None:
        capture_path = os.path.join(OUTPUT_DIR, f"{timestamp}_capture_{i:03d}.bmp")
        cv2.imwrite(capture_path, bmp_arr)

    print(f"{i+1}/{NUM_IMAGES} gray={gray_val}")


slm.close()

if camera:
    camera.EndAcquisition()
    camera.DeInit()

if cams:
    cams.Clear()

if system:
    system.ReleaseInstance()