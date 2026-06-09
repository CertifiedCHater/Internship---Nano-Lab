import os
import sys
import time
import numpy as np
from PIL import Image

OUTPUT_DIR       = r"C:\Users\mu00129\Desktop\slmnew10\Measurements2\PLUTO\capturesCameraNew1"
HOLOEYE_SDK_PATH = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.0.0\examples"
TEMP_BMP_PATH    = os.path.join(OUTPUT_DIR, "temp_sweep.bmp")

SLM_WIDTH        = 1920
SLM_HEIGHT       = 1080
LASER_WAVELENGTH = 633.0

EXPOSURE_US      = 85.0
WARMUP_FRAMES    = 20
SETTLE_TIME      = 0.5

CALIB_PREFIX     = "Capture_sweep_"
CALIB_SUFFIX     = ".bmp"
NUM_STEPS        = 1920   


def send_sweep_frame(step, slm, HEDSERR_NoError):
    img = np.zeros((SLM_HEIGHT, SLM_WIDTH), dtype=np.uint8)
    img[:, :step + 1] = 255                         # grow white from left

    Image.fromarray(img).save(TEMP_BMP_PATH)

    err, handle = slm.loadPhaseDataFromFile(TEMP_BMP_PATH)
    if err != HEDSERR_NoError:
        print(f"  loadPhaseDataFromFile failed at step {step}: {err}")
        return False

    err = handle.show()
    if err != HEDSERR_NoError:
        print(f"  handle.show() failed at step {step}: {err}")
        return False

    return True


def run_sweep(output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    sys.path.append(HOLOEYE_SDK_PATH)
    try:
        import HEDS
        from hedslib.heds_types import HEDSERR_NoError

        err = HEDS.SDK.Init(4, 0)
        assert err == HEDSERR_NoError, f"SDK Init failed: {err}"

        # Init PLUTO (index 0) — ERIS is left untouched
        slm = HEDS.SLM.Init("", True, 0.0)
        assert slm.errorCode() == HEDSERR_NoError, \
            f"SLM Init failed: {slm.errorCode()}"

        slm.setWavelength(LASER_WAVELENGTH)
        print(f"  PLUTO ready: {slm.width_px()} x {slm.height_px()} px")

    except Exception as e:
        print(f"  ERROR: SLM init failed: {e}")
        return

    try:
        import PySpin
        system = PySpin.System.GetInstance()
        cams   = system.GetCameras()
        assert cams.GetSize() > 0, "No camera found"

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

        print(f"  Discarding {WARMUP_FRAMES} warmup frames...")
        for _ in range(WARMUP_FRAMES):
            frm = camera.GetNextImage()
            frm.Release()
        print("  Camera ready.")

    except Exception as e:
        print(f"  ERROR: Camera init failed: {e}")
        HEDS.SDK.Close()
        return

    print(f"\n  Starting sweep: {NUM_STEPS} steps (full black → full white)...")
    failed = []

    for step in range(NUM_STEPS):          

        ok = send_sweep_frame(step, slm, HEDSERR_NoError)
        if not ok:
            failed.append(step)
            continue

        time.sleep(SETTLE_TIME)

        for _ in range(3):
            stale = camera.GetNextImage()
            stale.Release()

        try:
            raw = camera.GetNextImage()
            if not raw.IsIncomplete():
                frame    = raw.GetNDArray().astype(np.uint8)
                filename = f"{CALIB_PREFIX}{step:04d}{CALIB_SUFFIX}"
                Image.fromarray(frame).save(os.path.join(output_dir, filename))
            raw.Release()
        except Exception as e:
            print(f"  WARNING: Capture failed at step {step}: {e}")
            failed.append(step)

        if step % 100 == 0 or step == NUM_STEPS - 1:
            print(f"  [{step:4d}/{NUM_STEPS - 1}] done")

    try:
        camera.EndAcquisition()
        camera.DeInit()
        del camera
        cams.Clear()
        system.ReleaseInstance()
    except Exception as e:
        print(f"  Camera cleanup: {e}")

    try:
        HEDS.SDK.Close()
    except Exception as e:
        print(f"  SDK cleanup: {e}")

    if os.path.exists(TEMP_BMP_PATH):
        os.remove(TEMP_BMP_PATH)

    print(f"\n  Sweep complete.")
    print(f"  Saved to : {output_dir}")
    print(f"  Captured : {NUM_STEPS - len(failed)} / {NUM_STEPS}")
    if failed:
        print(f"  Failed   : {failed}")


if __name__ == "__main__":
    run_sweep(OUTPUT_DIR)
