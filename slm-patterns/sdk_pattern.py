import sys
import time
import numpy as np

# --- Holoeye SDK path ---
HOLOEYE_SDK_PATH = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.0.0\examples"
sys.path.append(HOLOEYE_SDK_PATH)

import HEDS
from hedslib.heds_types import HEDSERR_NoError


def display_grayscale_patch():

    # --- Init SDK and SLM ---
    err = HEDS.SDK.Init(4, 0)
    assert err == HEDSERR_NoError, f"SDK init failed: {err}"

    slm = HEDS.SLM.Init()
    assert slm.errorCode() == HEDSERR_NoError, f"SLM init failed: {slm.errorCode()}"

    # Auto-detect SLM resolution (same as pygame reading the monitor size)
    slm_width  = slm.width_px()
    slm_height = slm.height_px()
    print(f"SLM size: {slm_width} x {slm_height}")

    # --- Square settings (same as your pygame version) ---
    rect_width  = 300
    rect_height = 300

    # Position: slightly right of center (matching your rect_x=1000, rect_y=600)
    rect_x = 1000
    rect_y = 600
    print(f"rect_x: {rect_x}")
    print(f"rect_y: {rect_y}")

    gray_value  = 0
    last_update = time.time()

    print("Running — press Ctrl+C to stop")

    try:
        while True:
            current_time = time.time()

            # Update every second (same timing as your pygame version)
            if current_time - last_update >= 1:
                gray_value += 1
                if gray_value > 255:
                    gray_value = 0
                last_update = current_time
                print(f"  gray value: {gray_value}")

            # Build SLM image — black background, square at gray_value
            img = np.zeros((slm_height, slm_width), dtype=np.uint8)
            img[rect_y : rect_y + rect_height,
                rect_x : rect_x + rect_width] = gray_value

            # Send to SLM
            err, handle = slm.loadImageData(img)
            assert err == HEDSERR_NoError, f"Load error: {err}"
            err = handle.show()
            assert err == HEDSERR_NoError, f"Show error: {err}"

            time.sleep(1 / 60)   # ~60 fps, same as clock.tick(60)

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        slm.close()
        HEDS.SDK.Close()


display_grayscale_patch()
