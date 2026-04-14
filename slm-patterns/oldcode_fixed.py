import os
import sys
import time
import numpy as np
import cv2

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

OUTPUT_DIR = r"C:\Users\mu00129\Desktop\slm_patterns"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SLM_WIDTH = 1024
SLM_HEIGHT = 768
NUM_RANDOM_PATTERNS = 10

# -------------------------
# Initialize SDK and SLM
# -------------------------

err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

slm = HEDS.SLM.Init()
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# -------------------------
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
        print("No camera detected.")
        USE_CAMERA = False


# -------------------------
# Function to display pattern
# -------------------------

def display_pattern(pattern):

    err, dataHandle = slm.loadImageData(pattern)
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    err = dataHandle.show()
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    time.sleep(0.1)


def capture_image(name):

    if USE_CAMERA and camera:
        img = camera.GetNextImage()

        if not img.IsIncomplete():
            frame = img.GetNDArray()
            cv2.imwrite(os.path.join(OUTPUT_DIR, name), frame)

        img.Release()


# -------------------------
# Pattern 1: Black
# -------------------------

black = np.zeros((SLM_HEIGHT, SLM_WIDTH), dtype=np.uint8)

display_pattern(black)
capture_image("capture_black.bmp")

print("Displayed BLACK pattern")


# -------------------------
# Pattern 2: White
# -------------------------

white = np.full((SLM_HEIGHT, SLM_WIDTH), 255, dtype=np.uint8)

display_pattern(white)
capture_image("capture_white.bmp")

print("Displayed WHITE pattern")


# -------------------------
# Pattern 3: Gradient
# -------------------------

gradient = np.tile(
    np.linspace(0,255,SLM_WIDTH,dtype=np.uint8),
    (SLM_HEIGHT,1)
)

display_pattern(gradient)
capture_image("capture_gradient.bmp")

print("Displayed GRADIENT pattern")


# -------------------------
# Pattern 4: Vertical stripes
# -------------------------

stripes = np.zeros((SLM_HEIGHT, SLM_WIDTH), dtype=np.uint8)

for i in range(0, SLM_WIDTH, 40):
    stripes[:, i:i+20] = 255

display_pattern(stripes)
capture_image("capture_stripes.bmp")

print("Displayed STRIPES pattern")


# -------------------------
# Random patterns
# -------------------------

for i in range(NUM_RANDOM_PATTERNS):

    pattern = np.random.randint(
        0,256,
        (SLM_HEIGHT,SLM_WIDTH),
        dtype=np.uint8
    )

    display_pattern(pattern)

    capture_image(f"capture_random_{i:03d}.bmp")

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"slm_pattern_{i:03d}.bmp"),
        pattern
    )

    print(f"Random pattern {i+1}/{NUM_RANDOM_PATTERNS}")


# -------------------------
# Cleanup
# -------------------------

if USE_CAMERA and camera:
    camera.EndAcquisition()
    camera.DeInit()
    del camera

if USE_CAMERA and cams:
    cams.Clear()
    del cams

if USE_CAMERA and system:
    system.ReleaseInstance()

slm.close()
HEDS.SDK.Close()
