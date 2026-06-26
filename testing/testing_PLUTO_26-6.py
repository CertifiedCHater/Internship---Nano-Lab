from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image


OUTPUT_DIR = r"C:\Users\mu00129\Desktop\slmnew10\Measurements2\PLUTO\quicktest"


SLM_WIDTH  = 1920
SLM_HEIGHT = 1080


RECT_ROW_START = 400
RECT_ROW_END   = 680
RECT_COL_START = 150     
RECT_COL_END   = 450     


HOLOEYE_SDK_PATH = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.0.0\examples"
LASER_WAVELENGTH = 633.0
SETTLE_TIME      = 2.0    # seconds after each pattern (ERIS was 0.1; Pluto needs more)
FLUSH_FRAMES     = 8      # stale frames to discard after each pattern change
WARMUP_FRAMES    = 20     # duplicate frames after BeginAcquisition()
EXPOSURE_US      = 85.0


TEST_GRAYS = [0, 64, 128, 192, 255]


USE_PHASE_DATA = True

TEMP_BMP = "_qt_pattern.bmp"



def build_pattern(gray_val: int) -> np.ndarray:
    img = np.zeros((SLM_HEIGHT, SLM_WIDTH), dtype=np.uint8)
    img[RECT_ROW_START:RECT_ROW_END, RECT_COL_START:RECT_COL_END] = gray_val
    return img


def send_to_slm(slm, gray_val, HEDSERR_NoError, tmp_path) -> bool:
    Image.fromarray(build_pattern(gray_val)).save(tmp_path)
    if USE_PHASE_DATA:
        err, handle = slm.loadPhaseDataFromFile(tmp_path)
    else:
        err, handle = slm.loadImageDataFromFile(tmp_path)
    if err != HEDSERR_NoError:
        print(f"    load failed at gray {gray_val}: {err}")
        return False
    err = handle.show()
    if err != HEDSERR_NoError:
        print(f"    show() failed at gray {gray_val}: {err}")
        return False
    return True


def init_slm():
    sys.path.append(HOLOEYE_SDK_PATH)
    import HEDS
    from hedslib.heds_types import HEDSERR_NoError
    err = HEDS.SDK.Init(4, 0)
    assert err == HEDSERR_NoError, f"SDK Init failed: {err}"
    slm = HEDS.SLM.Init("", True, 0.0)
    assert slm.errorCode() == HEDSERR_NoError, f"SLM Init failed: {slm.errorCode()}"
    slm.setWavelength(LASER_WAVELENGTH)
    # ---- SANITY: confirm WHICH SLM this is. Pluto should report ~1920x1080.
    w, h = slm.width_px(), slm.height_px()
    print(f"  SLM reports {w} x {h} px")
    if (w, h) != (SLM_WIDTH, SLM_HEIGHT):
        print(f"  ** WARNING: SDK size {w}x{h} != your CONFIG {SLM_WIDTH}x{SLM_HEIGHT}.")
        print(f"     Either you're addressing the wrong SLM (ERIS instead of Pluto),")
        print(f"     or SLM_WIDTH/SLM_HEIGHT need correcting. Fix before trusting results.")
    return HEDS, HEDSERR_NoError, slm


def init_camera():
    import PySpin
    system = PySpin.System.GetInstance()
    cams = system.GetCameras()
    assert cams.GetSize() > 0, "No camera found"
    camera = cams.GetByIndex(0)
    camera.Init()
    nm = camera.GetNodeMap()
    PySpin.CEnumerationPtr(nm.GetNode("ExposureAuto")).SetIntValue(
        PySpin.CEnumerationPtr(nm.GetNode("ExposureAuto")).GetEntryByName("Off").GetValue())
    PySpin.CFloatPtr(nm.GetNode("ExposureTime")).SetValue(EXPOSURE_US)
    PySpin.CEnumerationPtr(nm.GetNode("AcquisitionMode")).SetIntValue(
        PySpin.CEnumerationPtr(nm.GetNode("AcquisitionMode")).GetEntryByName("Continuous").GetValue())
    camera.BeginAcquisition()
    for _ in range(WARMUP_FRAMES):
        camera.GetNextImage().Release()
    return PySpin, system, cams, camera


def grab_clean_frame(camera, flush=FLUSH_FRAMES) -> np.ndarray:
    for _ in range(flush):
        camera.GetNextImage().Release()
    raw = camera.GetNextImage()
    frame = raw.GetNDArray().astype(np.uint8) if not raw.IsIncomplete() else None
    raw.Release()
    return frame



def mode_watch():
    print("WATCH mode: flipping square between gray 0 and 255. Ctrl-C to stop.")
    HEDS, HEDSERR_NoError, slm = init_slm()
    tmp = os.path.join(os.path.dirname(TEMP_BMP) or ".", TEMP_BMP)
    try:
        toggle = 0
        while True:
            g = 0 if toggle else 255
            send_to_slm(slm, g, HEDSERR_NoError, tmp)
            print(f"  showing gray {g:3d}  (square cols {RECT_COL_START}-{RECT_COL_END})")
            toggle ^= 1
            time.sleep(SETTLE_TIME)
    except KeyboardInterrupt:
        print("\n  stopped.")
    finally:
        slm.close(); HEDS.SDK.Close()
        if os.path.exists(tmp):
            os.remove(tmp)



def mode_capture(output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    HEDS, HEDSERR_NoError, slm = init_slm()
    PySpin, system, cams, camera = init_camera()
    print(f"  capturing gray levels {TEST_GRAYS} ...")
    tmp = os.path.join(output_dir, TEMP_BMP)
    try:
        for g in TEST_GRAYS:
            if not send_to_slm(slm, g, HEDSERR_NoError, tmp):
                continue
            time.sleep(SETTLE_TIME)
            frame = grab_clean_frame(camera)
            if frame is not None:
                Image.fromarray(frame).save(
                    os.path.join(output_dir, f"Capture_gray_{g:03d}.bmp"))
                print(f"    gray {g:3d} -> saved")
    finally:
        try:
            camera.EndAcquisition(); camera.DeInit(); del camera
            cams.Clear(); system.ReleaseInstance()
        except Exception:
            pass
        slm.close(); HEDS.SDK.Close()
        if os.path.exists(tmp):
            os.remove(tmp)
    analyze_capture_set(output_dir, TEST_GRAYS)



def analyze_capture_set(folder, grays):
    """Decide if the square responded, and locate the Shift region on the camera."""
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    frames, used = [], []
    for g in grays:
        p = os.path.join(folder, f"Capture_gray_{g:03d}.bmp")
        if os.path.exists(p):
            frames.append(np.asarray(Image.open(p).convert("L"), dtype=float))
            used.append(g)
    if len(frames) < 2:
        print("  need at least 2 captured frames to compare.")
        return
    stack = np.stack(frames)                      # (n_gray, H, W)
    mean_img = stack.mean(0)
    std_img = stack.std(0)                        # per-pixel change across grays

    illum = mean_img > mean_img.max() * 0.30      # where the beam is
    if illum.sum() == 0:
        print("  no illuminated region found — beam/exposure problem, not phase.")
        return


    resp = std_img[illum].mean()
    resp_max = std_img[illum].max()
    diff = np.abs(frames[-1] - frames[0])
    print(f"  illuminated pixels       : {illum.sum():,}")
    print(f"  mean change in beam (std): {resp:.2f}  (max {resp_max:.1f})")
    print(f"  |gray{used[-1]} - gray{used[0]}| mean : {diff[illum].mean():.2f}")
    verdict = "RESPONDING ✓" if resp > 2.0 else "NOT responding ✗ (square is dead/in the dark)"
    print(f"  VERDICT                  : {verdict}")


    rows = np.where(illum.any(1))[0]
    r0, r1 = rows.min(), rows.max()
    col_var = std_img[r0:r1, :].mean(0)
    col_lit = (mean_img[r0:r1, :].mean(0) > mean_img.max() * 0.30)
    lit_cols = np.where(col_lit)[0]
    if lit_cols.size:
        var_in_lit = np.where(col_lit, col_var, np.nan)
        shift_c = int(np.nanargmax(var_in_lit))
        # NoShift = lit column with the LOWEST variance
        noshift_c = int(lit_cols[np.nanargmin(var_in_lit[lit_cols])])
        print("\n  Suggested ROIs for detect_roi / main script (rough, refine as needed):")
        print(f"    CAM_ROW_START  = {r0}")
        print(f"    CAM_ROW_END    = {r1}")
        print(f"    Shift region   ~ column {shift_c}  (highest variance = pattern lands here)")
        print(f"    NoShift region ~ column {noshift_c} (lit but static)")
        print(f"    KC             ~ {(r0 + r1) // 2}  (mid-row; any fixed row works)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(mean_img, cmap="gray"); ax[0].set_title("Mean frame (beam)")
        im = ax[1].imshow(std_img, cmap="inferno")
        ax[1].set_title("Change across gray levels (bright = Shift region)")
        fig.colorbar(im, ax=ax[1], fraction=0.046)
        for a in ax:
            a.axis("off")
        out = os.path.join(folder, "quicktest_heatmap.png")
        fig.tight_layout(); fig.savefig(out, dpi=130)
        print(f"\n  heatmap saved -> {out}")
    except Exception as e:
        print(f"  (heatmap skipped: {e})")


def main():
    ap = argparse.ArgumentParser(description="Pluto quick-test")
    ap.add_argument("--watch", action="store_true", help="flip 0<->255 forever (no camera)")
    ap.add_argument("--capture", action="store_true", help="capture a few grays then analyze")
    ap.add_argument("--analyze-only", action="store_true", help="analyze an existing folder")
    ap.add_argument("--dir", default=OUTPUT_DIR, help="folder for --analyze-only")
    args = ap.parse_args()

    if args.watch:
        mode_watch()
    elif args.analyze_only:
        analyze_capture_set(args.dir, TEST_GRAYS)
    else:                       # default: full capture + analyze
        mode_capture(args.dir)


if __name__ == "__main__":
    main()
