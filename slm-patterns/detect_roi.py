import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


IMAGE_PATHS = [
    r"C:\Users\mu00129\Desktop\slmnew10\capturesCameraSquareNew3\Capture_gray_000.bmp",
    r"C:\Users\mu00129\Desktop\slmnew10\capturesCameraSquareNew3\Capture_gray_050.bmp",
    r"C:\Users\mu00129\Desktop\slmnew10\capturesCameraSquareNew3\Capture_gray_100.bmp",
    r"C:\Users\mu00129\Desktop\slmnew10\capturesCameraSquareNew3\Capture_gray_150.bmp",
    r"C:\Users\mu00129\Desktop\slmnew10\capturesCameraSquareNew3\Capture_gray_200.bmp",
    r"C:\Users\mu00129\Desktop\slmnew10\capturesCameraSquareNew3\Capture_gray_254.bmp",
]

OUTPUT_PLOT = "roi_detection_result.png"


VARIANCE_PERCENTILE = 60

EDGE_MARGIN = 30




def load_images(paths):
    arrays = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  WARNING: File not found: {p}")
            continue
        arr = np.array(Image.open(p).convert("L"), dtype=float)
        arrays.append(arr)
        print(f"  Loaded: {os.path.basename(p)}  shape={arr.shape}")
    if not arrays:
        raise FileNotFoundError("No images loaded. Check IMAGE_PATHS.")
    return arrays


def compute_variance_map(arrays):
    """Pixel-wise variance across all images."""
    stack = np.stack(arrays, axis=0)
    return np.var(stack, axis=0)


def find_active_range(profile, percentile=VARIANCE_PERCENTILE, margin=EDGE_MARGIN):
    threshold = np.percentile(profile, percentile)
    active    = np.where(profile > threshold)[0]
    if len(active) == 0:
        raise ValueError("No active region found. Try lowering VARIANCE_PERCENTILE.")
    return int(active.min()) + margin, int(active.max()) - margin


def find_noshift_region(col_var, shift_c1, margin=EDGE_MARGIN):
    left_region = col_var[:shift_c1 - margin]
    if len(left_region) < 100:
        return None, None

    low_thresh = np.percentile(left_region, 40)
    candidate  = np.where(left_region > low_thresh)[0]

    if len(candidate) == 0:
        return None, None

    end   = min(candidate.max(), shift_c1 - 100)
    start = max(end - 600, candidate.min() + margin)
    return int(start), int(end)


def detect_roi(image_paths, variance_percentile=VARIANCE_PERCENTILE, margin=EDGE_MARGIN):
    print("\n=== ROI & KC Detection ===\n")
    arrays  = load_images(image_paths)
    var_map = compute_variance_map(arrays)

    row_var = var_map.mean(axis=1)   # variance profile along rows
    col_var = var_map.mean(axis=0)   # variance profile along columns

    row_start, row_end = find_active_range(row_var, variance_percentile, margin)

    shift_c1, shift_c2 = find_active_range(col_var, variance_percentile, margin)

    noshift_c1, noshift_c2 = find_noshift_region(col_var, shift_c1, margin)
    if noshift_c1 is None:
        noshift_c1 = max(50, shift_c1 - 700)
        noshift_c2 = shift_c1 - 100
        print("  WARNING: Could not auto-detect NoShift region. Using fallback.")

    roi_height = row_end - row_start
    kc         = roi_height // 2

    results = {
        "CAM_ROW_START"  : row_start,
        "CAM_ROW_END"    : row_end,
        "CAM_SHIFT_C1"   : shift_c1,
        "CAM_SHIFT_C2"   : shift_c2,
        "CAM_NOSHIFT_C1" : noshift_c1,
        "CAM_NOSHIFT_C2" : noshift_c2,
        "KC"             : kc,
        "ROI_HEIGHT"     : roi_height,
    }

    print("\n=== Recommended config values ===\n")
    print(f"  CAM_ROW_START   = {row_start}")
    print(f"  CAM_ROW_END     = {row_end}")
    print(f"  CAM_SHIFT_C1    = {shift_c1}   (left edge of rectangle)")
    print(f"  CAM_SHIFT_C2    = {shift_c2}   (right edge of rectangle)")
    print(f"  CAM_NOSHIFT_C1  = {noshift_c1}   (left reference region start)")
    print(f"  CAM_NOSHIFT_C2  = {noshift_c2}   (left reference region end)")
    print(f"  KC              = {kc}   (= ROI_HEIGHT / 2 = {roi_height} / 2)")
    print()
    print("  KC explanation:")
    print(f"    Your row ROI is {roi_height} pixels tall.")
    print(f"    KC = {kc} sits in the middle of the profile,")
    print(f"    well away from the edges where signal is weak.")
    print(f"    It must be the SAME value for every image in your sweep.")

    return results, arrays[0], row_var, col_var, var_map


def plot_results(results, ref_image, row_var, col_var, var_map, output_path):
    r0  = results["CAM_ROW_START"]
    r1  = results["CAM_ROW_END"]
    sc1 = results["CAM_SHIFT_C1"]
    sc2 = results["CAM_SHIFT_C2"]
    nc1 = results["CAM_NOSHIFT_C1"]
    nc2 = results["CAM_NOSHIFT_C2"]
    kc  = results["KC"]
    kc_abs = r0 + kc   # absolute row in camera image

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ROI & KC Auto-Detection Result", fontsize=13)

    # Panel 1: camera image with rectangles
    axes[0,0].imshow(ref_image, cmap="gray", vmin=0, vmax=255)
    axes[0,0].add_patch(Rectangle(
        (sc1, r0), sc2-sc1, r1-r0,
        linewidth=2, edgecolor="#E24B4A", facecolor="none", label="Shift (rectangle)"
    ))
    axes[0,0].add_patch(Rectangle(
        (nc1, r0), nc2-nc1, r1-r0,
        linewidth=2, edgecolor="#378ADD", facecolor="none", label="NoShift (reference)"
    ))
    axes[0,0].axhline(kc_abs, color="#3B6D11", linewidth=1.5,
                      linestyle=":", label=f"KC row = {kc_abs} (kc={kc})")
    axes[0,0].legend(fontsize=8, loc="upper left")
    axes[0,0].set_title("Camera image — detected ROIs")
    axes[0,0].axis("off")

    axes[0,1].imshow(var_map, cmap="hot")
    axes[0,1].set_title("Pixel variance map\n(bright = changes between images)")
    axes[0,1].axis("off")

    axes[1,0].plot(row_var, linewidth=0.8)
    axes[1,0].axvline(r0,      color="#E24B4A", linestyle="--",
                      label=f"CAM_ROW_START = {r0}")
    axes[1,0].axvline(r1,      color="#E24B4A", linestyle="--",
                      label=f"CAM_ROW_END   = {r1}")
    axes[1,0].axvline(kc_abs,  color="#3B6D11", linestyle=":",
                      label=f"KC abs row    = {kc_abs}  →  kc={kc}")
    axes[1,0].set_xlabel("Row (y)")
    axes[1,0].set_ylabel("Mean variance across columns")
    axes[1,0].set_title("Row variance — determines ROW_START, ROW_END, KC")
    axes[1,0].legend(fontsize=8)

    axes[1,1].plot(col_var, linewidth=0.8)
    axes[1,1].axvline(sc1, color="#E24B4A", linestyle="--",
                      label=f"CAM_SHIFT_C1   = {sc1}")
    axes[1,1].axvline(sc2, color="#E24B4A", linestyle="--",
                      label=f"CAM_SHIFT_C2   = {sc2}")
    axes[1,1].axvline(nc1, color="#378ADD", linestyle="--",
                      label=f"CAM_NOSHIFT_C1 = {nc1}")
    axes[1,1].axvline(nc2, color="#378ADD", linestyle="--",
                      label=f"CAM_NOSHIFT_C2 = {nc2}")
    axes[1,1].set_xlabel("Column (x)")
    axes[1,1].set_ylabel("Mean variance across rows")
    axes[1,1].set_title("Column variance — determines SHIFT cols and NoShift cols")
    axes[1,1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.show()
    print(f"\n  Plot saved to: {output_path}")

if __name__ == "__main__":
    results, ref_img, row_var, col_var, var_map = detect_roi(IMAGE_PATHS)
    plot_results(results, ref_img, row_var, col_var, var_map, OUTPUT_PLOT)

    print("\n=== Copy these into your calibration script Section 1 ===\n")
    print(f"CAM_ROW_START   = {results['CAM_ROW_START']}")
    print(f"CAM_ROW_END     = {results['CAM_ROW_END']}")
    print(f"CAM_NOSHIFT_C1  = {results['CAM_NOSHIFT_C1']}")
    print(f"CAM_NOSHIFT_C2  = {results['CAM_NOSHIFT_C2']}")
    print(f"CAM_SHIFT_C1    = {results['CAM_SHIFT_C1']}")
    print(f"CAM_SHIFT_C2    = {results['CAM_SHIFT_C2']}")
    print(f"KC              = {results['KC']}")
