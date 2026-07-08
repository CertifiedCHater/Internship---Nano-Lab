import os
import sys
import time
from datetime import datetime
import numpy as np
from PIL import Image

OUTPUT_DIR = r"C:\Users\mu00129\Desktop\slmnew10\Measurements2\PLUTO\timelapse"
HOLOEYE_SDK_PATH = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.0.0\examples"
LASER_WAVELENGTH = 633.0

DURATION_MIN = 20          
INTERVAL_SEC = 1.0         
EXPOSURE_US  = 85.0
WARMUP_FRAMES = 20
FLUSH_FRAMES  = 3
USE_PHASE_DATA = True      
GRAY = 255                 

TMP_BMP = os.path.join(OUTPUT_DIR, "_tmp_white.bmp")


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sys.path.append(HOLOEYE_SDK_PATH)
    import HEDS
    from hedslib.heds_types import HEDSERR_NoError

    assert HEDS.SDK.Init(4, 0) == HEDSERR_NoError, "SDK Init failed"
    slm = HEDS.SLM.Init("", True, 0.0)
    assert slm.errorCode() == HEDSERR_NoError, "SLM Init failed"
    slm.setWavelength(LASER_WAVELENGTH)
    w, h = slm.width_px(), slm.height_px()
    Image.fromarray(np.full((h, w), GRAY, np.uint8)).save(TMP_BMP)
    err, handle = (slm.loadPhaseDataFromFile(TMP_BMP) if USE_PHASE_DATA
                   else slm.loadImageDataFromFile(TMP_BMP))
    assert err == HEDSERR_NoError, f"load failed: {err}"
    handle.show()
    print(f"  PLUTO {w}x{h} at full white (gray {GRAY})")

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
    print("  camera ready")

    duration = DURATION_MIN * 60
    n_expected = int(duration / INTERVAL_SEC)
    print(f"  capturing 1 frame / {INTERVAL_SEC:.0f}s for {DURATION_MIN} min "
          f"(~{n_expected} frames)")

    log_path = os.path.join(OUTPUT_DIR, "timelapse_log.csv")
    start = time.time()
    i = 0
    try:
        with open(log_path, "w") as log:
            log.write("index,filename,timestamp,elapsed_sec\n")
            while time.time() - start < duration:
                # capture one clean frame
                for _ in range(FLUSH_FRAMES):
                    cam.GetNextImage().Release()
                raw = cam.GetNextImage()
                frame = raw.GetNDArray().astype(np.uint8) if not raw.IsIncomplete() else None
                raw.Release()

                now = datetime.now()
                ts = now.strftime("%Y%m%d_%H%M%S")
                elapsed = time.time() - start
                fname = f"Capture_{i:05d}_{ts}.bmp"
                if frame is not None:
                    Image.fromarray(frame).save(os.path.join(OUTPUT_DIR, fname))
                    log.write(f"{i},{fname},{now.isoformat()},{elapsed:.2f}\n")
                    log.flush()
                if i % 60 == 0:
                    print(f"  [{i:5d}]  t = {elapsed/60:5.1f} min")
                i += 1
                # keep a steady cadence (schedule the next frame at start + i*interval)
                time.sleep(max(0.0, start + i * INTERVAL_SEC - time.time()))
    except KeyboardInterrupt:
        print("\n  stopped early by user.")
    finally:
        try:
            cam.EndAcquisition(); cam.DeInit(); del cam
            cams.Clear(); system.ReleaseInstance()
        except Exception as e:
            print(f"  camera cleanup: {e}")
        try:
            HEDS.SDK.Close()
        except Exception as e:
            print(f"  SDK cleanup: {e}")
        if os.path.exists(TMP_BMP):
            os.remove(TMP_BMP)
        print(f"  done: {i} frames saved to {OUTPUT_DIR}")
        print(f"  log: {log_path}")


if __name__ == "__main__":
    run()
