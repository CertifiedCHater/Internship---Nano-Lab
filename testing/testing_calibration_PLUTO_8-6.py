import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import signal



OUTPUT_DIR = r"C:\Users\mu00129\Desktop\slmnew10\Measurements2\PLUTO\capturesCameraNew1"
CALIB_DIR  = OUTPUT_DIR

SLM_WIDTH  = 1920
SLM_HEIGHT = 1200

RECT_ROW_START = 450
RECT_ROW_END   = 750
RECT_COL_START = 810
RECT_COL_END   = 1110


CAM_ROW_START  = 266
CAM_ROW_END    = 1130

CAM_NOSHIFT_C1 = 33
CAM_NOSHIFT_C2 = 449

CAM_SHIFT_C1   = 549
CAM_SHIFT_C2   = 1310

KC             = 432


NUM_GRAY_LEVELS   = 256
CALIB_PREFIX      = "Capture_gray_"
CALIB_SUFFIX      = ".bmp"
CALIB_RANGE_START = 0
CALIB_RANGE_END   = 255


HOLOEYE_SDK_PATH = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.0.0\examples"
LASER_WAVELENGTH = 633.0
SETTLE_TIME      = 1
WARMUP_FRAMES    = 20
EXPOSURE_US      = 85.0


TEMP_BMP = os.path.join(OUTPUT_DIR, "_temp_pattern.bmp")

# def build_slm_pattern(gray_val):
#     img = np.zeros((SLM_HEIGHT, SLM_WIDTH), dtype=np.uint8)
#     img[RECT_ROW_START:RECT_ROW_END,
#         RECT_COL_START:RECT_COL_END] = gray_val
#     return img

def send_to_slm(gray_val, slm, HEDSERR_NoError):
    # Build grayscale pattern — same as before
    img = np.zeros((SLM_HEIGHT, SLM_WIDTH), dtype=np.uint8)
    img[RECT_ROW_START:RECT_ROW_END,
        RECT_COL_START:RECT_COL_END] = gray_val

    Image.fromarray(img).save(TEMP_BMP)

    err, handle = slm.loadPhaseDataFromFile(TEMP_BMP)
    if err != HEDSERR_NoError:
        print(f"  loadPhaseDataFromFile failed at gray {gray_val}: {err}")
        return False

    err = handle.show()
    if err != HEDSERR_NoError:
        print(f"  handle.show() failed at gray {gray_val}: {err}")
        return False

    return True


def find_central_frequency(L_s):
    N    = len(L_s)
    half = np.abs(np.fft.fft(L_s)[:N // 2])
    skip = 14
    if half.size <= skip:
        raise ValueError("Signal too short.")
    return (int(np.argmax(half[skip:])) + skip) / N * 2.0


def find_phi(I, kc=KC, band_width=0.015, env_frac=1/6, mute_first=200):
    """
    Extract phase from a 1D interferometric intensity profile.
    Based on Shen & Wang (2005) / Ma & Wang (2013).
    kc must be the same for every image in the sweep.
    """
    I   = np.asarray(I, dtype=float).ravel()
    N   = I.size
    k   = np.arange(N, dtype=float)
    L_s = I - I.mean()
    CF  = find_central_frequency(L_s)
    Mf  = int(round(N / 4))
    if Mf % 2 == 1:
        Mf += 1
    numtaps = Mf + 1
    eps  = np.finfo(float).eps ** 0.5
    bw   = float(band_width)
    CF   = float(np.clip(CF, bw + eps, 1.0 - bw - eps))
    low  = max(CF - bw, eps)
    high = min(CF + bw, 1.0 - eps)
    b    = signal.firwin(numtaps, [low, high], pass_zero=False)
    L_s  = signal.lfilter(b, [1.0], L_s)
    L_a  = signal.hilbert(L_s)
    nd   = Mf // 2
    L_s  = L_s[nd:];  L_a = L_a[nd:];  k = k[:L_s.size]
    evlp = np.abs(L_a)
    if 0 < mute_first < evlp.size:
        evlp[:mute_first] = 0.0
    idx_eff = np.where(evlp > evlp.max() * env_frac)[0]
    if idx_eff.size == 0:
        raise ValueError("No valid data in phase extraction.")
    N_begin = max(idx_eff[0], Mf - nd)
    N_end   = idx_eff[-1]
    L_s = L_s[N_begin:N_end+1]
    L_a = L_a[N_begin:N_end+1]
    k   = k[N_begin:N_end+1]
    abs_La = np.abs(L_a);  abs_La[abs_La == 0] = 1.0
    L_s = L_s / abs_La;    L_a = L_a / abs_La
    phi_n = np.unwrap(np.arctan2(np.imag(L_a), np.real(L_a)) * 2.0) / 2.0
    if np.real(L_a[0]) < 0:
        phi_n = phi_n - np.pi
    A = np.vstack([k, np.ones_like(k)]).T
    est, *_ = np.linalg.lstsq(A, phi_n, rcond=None)
    L_est, phi0 = est
    phi0 += 2 * np.pi * (-np.round(phi0 / (2 * np.pi)) - 1.0)
    return float(L_est * kc + phi0)


def get_profiles(image_array):
    noshift = np.mean(
        image_array[CAM_ROW_START:CAM_ROW_END,
                    CAM_NOSHIFT_C1:CAM_NOSHIFT_C2], axis=1
    )
    shift = np.mean(
        image_array[CAM_ROW_START:CAM_ROW_END,
                    CAM_SHIFT_C1:CAM_SHIFT_C2], axis=1
    )
    return noshift, shift



def run_capture(output_dir = OUTPUT_DIR):
    print("=" * 60)
    print("SECTION 3: LIVE CAPTURE")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)


    assert RECT_COL_END <= SLM_WIDTH, \
        f"Rectangle too wide: {RECT_COL_END} > {SLM_WIDTH}"
    assert RECT_ROW_END <= SLM_HEIGHT, \
        f"Rectangle too tall: {RECT_ROW_END} > {SLM_HEIGHT}"

    sys.path.append(HOLOEYE_SDK_PATH)
    try:
        import HEDS
        from hedslib.heds_types import HEDSERR_NoError

        err = HEDS.SDK.Init(4, 0)
        assert err == HEDSERR_NoError, f"SDK Init failed: {err}"

        slm = HEDS.SLM.Init("", True, 0.0)
        assert slm.errorCode() == HEDSERR_NoError, \
            f"SLM Init failed: {slm.errorCode()}"

        slm.setWavelength(LASER_WAVELENGTH)
        print(f"  SLM ready: {slm.width_px()} x {slm.height_px()} px")

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

    print(f"\n  Capturing {NUM_GRAY_LEVELS} gray levels...")
    failed = []

    # for gray_val in range(NUM_GRAY_LEVELS):

    #     img = build_slm_pattern(gray_val)
    #     Image.fromarray(img).save(TEMP_BMP)

    #     err, handle = slm.loadImageDataFromFile(TEMP_BMP)
    #     if err != HEDSERR_NoError:
    #         print(f"  WARNING: loadImageDataFromFile failed at gray {gray_val}: {err}")
    #         failed.append(gray_val)
    #         continue

    #     err = handle.show()
    #     if err != HEDSERR_NoError:
    #         print(f"  WARNING: handle.show() failed at gray {gray_val}: {err}")
    #         failed.append(gray_val)
    #         continue

    #     time.sleep(SETTLE_TIME)

    #     try:
    #         raw = camera.GetNextImage()
    #         if not raw.IsIncomplete():
    #             frame    = raw.GetNDArray().astype(np.uint8)
    #             filename = f"{CALIB_PREFIX}{gray_val:03d}{CALIB_SUFFIX}"
    #             Image.fromarray(frame).save(
    #                 os.path.join(output_dir, filename)
    #             )
    #         raw.Release()
    #     except Exception as e:
    #         print(f"  WARNING: Capture failed at gray {gray_val}: {e}")
    #         failed.append(gray_val)

    #     if gray_val % 32 == 0 or gray_val == 255:
    #         print(f"  [{gray_val:3d}/255] done")




    for gray_val in range(NUM_GRAY_LEVELS):

        ok = send_to_slm(gray_val, slm, HEDSERR_NoError)
        if not ok:
            failed.append(gray_val)
            continue

        time.sleep(SETTLE_TIME)

    # Flush stale frames
        for _ in range(2):
            stale = camera.GetNextImage()
            stale.Release()

    # Capture clean frame
        try:
            raw = camera.GetNextImage()
            if not raw.IsIncomplete():
                frame    = raw.GetNDArray().astype(np.uint8)
                filename = f"{CALIB_PREFIX}{gray_val:03d}{CALIB_SUFFIX}"
                Image.fromarray(frame).save(os.path.join(output_dir, filename))
            raw.Release()
        except Exception as e:
            print(f"  WARNING: Capture failed at gray {gray_val}: {e}")
            failed.append(gray_val)

        if gray_val % 32 == 0 or gray_val == 255:
            print(f"  [{gray_val:3d}/255] done")

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

    if os.path.exists(TEMP_BMP):
        os.remove(TEMP_BMP)

    print(f"\n  Capture complete.")
    print(f"  Saved to  : {output_dir}")
    print(f"  Captured  : {NUM_GRAY_LEVELS - len(failed)} / {NUM_GRAY_LEVELS}")
    if failed:
        print(f"  Failed    : {failed}")



def run_calibration_check(calib_dir=CALIB_DIR):
    print("=" * 60)
    print("SECTION 4: CALIBRATION CHECK")
    print("=" * 60)
    print(f"  Loading from  : {calib_dir}")
    print(f"  NoShift cols  : {CAM_NOSHIFT_C1} to {CAM_NOSHIFT_C2}")
    print(f"  Shift cols    : {CAM_SHIFT_C1} to {CAM_SHIFT_C2}")
    print(f"  Row range     : {CAM_ROW_START} to {CAM_ROW_END}")
    print(f"  KC            : {KC}\n")

    gray_range  = range(CALIB_RANGE_START, CALIB_RANGE_END + 1)
    phase_diffs = []
    missing     = []

    for i in gray_range:
        fname = f"{CALIB_PREFIX}{i:03d}{CALIB_SUFFIX}"
        fpath = os.path.join(calib_dir, fname)

        if not os.path.exists(fpath):
            print(f"  WARNING: Missing {fname}")
            missing.append(i)
            phase_diffs.append(np.nan)
            continue

        arr = np.array(Image.open(fpath).convert("L"), dtype=float)
        noshift, shift = get_profiles(arr)

        try:
            phi_ns = find_phi(noshift, kc=KC)
            phi_s  = find_phi(shift,   kc=KC)
            phase_diffs.append(phi_s - phi_ns)
        except Exception as e:
            print(f"  WARNING: Phase extraction failed at gray {i:03d}: {e}")
            phase_diffs.append(np.nan)

    change     = np.array(phase_diffs)
    valid_mask = ~np.isnan(change)

    change[valid_mask] = np.unwrap(change[valid_mask], period=np.pi)
    first_valid = np.where(valid_mask)[0][0]
    change -= change[first_valid]
    last_valid = np.where(valid_mask)[0][-1]
    if change[last_valid] < 0:
        change = -change

    change_pi = change / np.pi
    gray_axis = np.array(list(gray_range))

    valid_gv  = gray_axis[valid_mask]
    valid_phi = change_pi[valid_mask]
    coeffs    = np.polyfit(valid_gv, valid_phi, 1)
    ideal     = np.polyval(coeffs, gray_axis)

    deviation = change_pi - ideal
    max_dev   = np.nanmax(np.abs(deviation))

    slope = coeffs[0]
    print(f"  Measured phase range : {slope * 255:.3f}π  (slope={slope:.5f}π/gray)")
    print(f"  Max deviation        : {max_dev:.4f}π")
    print(f"  Requirement          : < 0.1π")
    print(f"  Status               : {'PASS ✓' if max_dev < 0.1 else 'FAIL ✗'}")
    if missing:
        print(f"  Missing files        : {missing}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("SLM Calibration Check", fontsize=13)

    axes[0].plot(gray_axis, change_pi, label="Measured phase", linewidth=1)
    axes[0].plot(gray_axis, ideal, label=f"Linear fit ({slope*255:.2f}π range)",
                 linestyle="--", color="orange")
    axes[0].set_xlabel("Gray Scale [8-bit]")
    axes[0].set_ylabel("Phase Shift [Rad/π]")
    axes[0].set_title("Phase vs Gray Scale")
    axes[0].legend()

    axes[1].plot(gray_axis, deviation, linewidth=1)
    axes[1].axhline( 0.1, color="r", linestyle="--", label="+0.1π limit")
    axes[1].axhline(-0.1, color="r", linestyle="--", label="-0.1π limit")
    axes[1].set_xlabel("Gray Scale [8-bit]")
    axes[1].set_ylabel("Deviation [Rad/π]")
    axes[1].set_title(f"Deviation from linear fit  (max={max_dev:.4f}π)")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(calib_dir, "calibration_result.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\n  Plot saved to: {out_path}")

    return change_pi, deviation, max_dev



# def show_figure(image_path_ref, image_path_shifted=None):
#     arr_ref = np.array(Image.open(image_path_ref).convert("L"), dtype=float)
#     noshift_ref, shift_ref = get_profiles(arr_ref)
#     rows = np.arange(CAM_ROW_START, CAM_ROW_END)
#
#     fig, axes = plt.subplots(1, 2, figsize=(13, 5))
#     fig.suptitle("Camera image + interference profiles", fontsize=12)
#
#     axes[0].imshow(arr_ref, cmap="gray", vmin=0, vmax=255)
#     axes[0].add_patch(Rectangle(
#         (CAM_NOSHIFT_C1, CAM_ROW_START),
#         CAM_NOSHIFT_C2 - CAM_NOSHIFT_C1,
#         CAM_ROW_END - CAM_ROW_START,
#         linewidth=2, edgecolor="#378ADD", facecolor="none", label="NoShift"
#     ))
#     axes[0].add_patch(Rectangle(
#         (CAM_SHIFT_C1, CAM_ROW_START),
#         CAM_SHIFT_C2 - CAM_SHIFT_C1,
#         CAM_ROW_END - CAM_ROW_START,
#         linewidth=2, edgecolor="#E24B4A", facecolor="none", label="Shift"
#     ))
#     axes[0].set_title("Camera image + ROI rectangles")
#     axes[0].legend(loc="upper right", fontsize=9)
#     axes[0].axis("off")
#
#     axes[1].plot(rows, noshift_ref, color="#378ADD", linewidth=1,
#                  label="NoShift — gray 000")
#     axes[1].plot(rows, shift_ref,   color="#E24B4A", linewidth=1,
#                  label="Shift — gray 000")
#
#     if image_path_shifted:
#         arr_s = np.array(Image.open(image_path_shifted).convert("L"), dtype=float)
#         noshift_s, shift_s = get_profiles(arr_s)
#         axes[1].plot(rows, noshift_s, color="#378ADD", linewidth=1,
#                      linestyle="--", alpha=0.6, label="NoShift — gray 128")
#         axes[1].plot(rows, shift_s,   color="#E24B4A", linewidth=1,
#                      linestyle="--", alpha=0.6, label="Shift — gray 128")
#
#     axes[1].set_xlabel("Row (y)")
#     axes[1].set_ylabel("Intensity")
#     axes[1].set_title("Averaged interference fringes on camera")
#     axes[1].legend(fontsize=8)
#
#     plt.tight_layout()
#     plt.savefig("figure_replica.png", dpi=150)
#     plt.show()
#     print("  Saved figure_replica.png")


if __name__ == "__main__":

    run_capture(OUTPUT_DIR)

    run_calibration_check(CALIB_DIR)

    # show_figure(
    #     os.path.join(OUTPUT_DIR, "Capture_gray_000.bmp"),
    #     os.path.join(OUTPUT_DIR, "Capture_gray_128.bmp")
    # )
