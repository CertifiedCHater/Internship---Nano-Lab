import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.ndimage import uniform_filter1d

IMAGE_PATHS = [
    r"C:\Users\mu00129\Desktop\slmnew10\capturesCameraSquareNew3\Capture_gray_000.bmp",
    r"C:\Users\mu00129\Desktop\slmnew10\capturesCameraSquareNew3\Capture_gray_050.bmp",
    r"C:\Users\mu00129\Desktop\slmnew10\capturesCameraSquareNew3\Capture_gray_100.bmp",
    r"C:\Users\mu00129\Desktop\slmnew10\capturesCameraSquareNew3\Capture_gray_150.bmp",
    r"C:\Users\mu00129\Desktop\slmnew10\capturesCameraSquareNew3\Capture_gray_200.bmp",
    r"C:\Users\mu00129\Desktop\slmnew10\capturesCameraSquareNew3\Capture_gray_254.bmp",
]
OUTPUT_PLOT = "roi_detection_result.png"
ACTIVE_FRAC = 0.25           
EDGE_MARGIN = 30
LIT_FRACTION = 0.30          
NOSHIFT_WIDTH = 400          
SMOOTH_WIN = 41              
                             


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
    """Pixel-wise variance across all images (bright = changes between images)."""
    return np.var(np.stack(arrays, axis=0), axis=0)


def compute_mean_map(arrays):
    """Pixel-wise mean across all images (bright = illuminated)."""
    return np.mean(np.stack(arrays, axis=0), axis=0)


def find_active_range(profile, frac=ACTIVE_FRAC, margin=EDGE_MARGIN):
    prof = uniform_filter1d(profile.astype(float), size=SMOOTH_WIN, mode="nearest")
    p = prof - prof.min()
    peak = p.max()
    if peak <= 0:
        raise ValueError("Flat profile — no active region (is the SLM modulating?).")
    idx = np.where(p > frac * peak)[0]
    if len(idx) == 0:
        raise ValueError("No active region found; lower ACTIVE_FRAC.")
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)   # contiguous blocks
    run = max(runs, key=len)                                  # the dominant one
    lo, hi = int(run.min()) + margin, int(run.max()) - margin
    if hi <= lo:                                              # narrower than 2*margin
        lo, hi = int(run.min()), int(run.max())
    return lo, hi


def find_noshift_region(col_var, col_mean, shift_c1, shift_c2,
                        margin=EDGE_MARGIN, width=NOSHIFT_WIDTH):
    lit    = col_mean > col_mean.max() * LIT_FRACTION          # has light
    static = col_var  < np.percentile(col_var, 40)             # doesn't change
    good = lit & static
    good[max(0, shift_c1 - margin):shift_c2 + margin] = False   # exclude the square
    idx = np.where(good)[0]
    if len(idx) == 0:
        return None, None
    left = idx[idx < shift_c1]                                  # prefer left of the square
    chosen = left if len(left) >= 50 else idx
    c1 = int(chosen.min()) + margin
    c2 = min(int(chosen.max()) - margin, c1 + width)
    if c2 <= c1:
        return None, None
    return c1, c2


def detect_roi(image_paths, margin=EDGE_MARGIN):
    print("\n=== ROI & KC Detection ===\n")
    arrays  = load_images(image_paths)
    var_map = compute_variance_map(arrays)
    mean_map = compute_mean_map(arrays)
    row_var = var_map.mean(axis=1)
    col_var = var_map.mean(axis=0)
    col_mean = mean_map.mean(axis=0)                            # brightness per column

    row_start, row_end = find_active_range(row_var, margin=margin)
    shift_c1, shift_c2 = find_active_range(col_var, margin=margin)
    noshift_c1, noshift_c2 = find_noshift_region(col_var, col_mean, shift_c1, shift_c2, margin)
    if noshift_c1 is None:
        noshift_c1 = max(50, shift_c1 - 700)
        noshift_c2 = shift_c1 - 100
        print("  WARNING: Could not auto-detect a lit+static NoShift region. Using fallback.")

    roi_height = row_end - row_start
    kc = roi_height // 2
    results = {
        "CAM_ROW_START": row_start, "CAM_ROW_END": row_end,
        "CAM_SHIFT_C1": shift_c1, "CAM_SHIFT_C2": shift_c2,
        "CAM_NOSHIFT_C1": noshift_c1, "CAM_NOSHIFT_C2": noshift_c2,
        "KC": kc, "ROI_HEIGHT": roi_height,
    }

    print("\n=== Recommended config values ===\n")
    for klab, v in results.items():
        if klab != "ROI_HEIGHT":
            print(f"  {klab:15s} = {v}")
    print(f"\n  KC = ROI_HEIGHT/2 = {roi_height}/2 = {kc}  (mid-profile; must be the "
          f"same for every image in the sweep)")
    return results, arrays[0], row_var, col_var, var_map


def plot_results(results, ref_image, row_var, col_var, var_map, output_path):
    r0, r1 = results["CAM_ROW_START"], results["CAM_ROW_END"]
    sc1, sc2 = results["CAM_SHIFT_C1"], results["CAM_SHIFT_C2"]
    nc1, nc2 = results["CAM_NOSHIFT_C1"], results["CAM_NOSHIFT_C2"]
    kc = results["KC"]; kc_abs = r0 + kc

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ROI & KC Auto-Detection Result", fontsize=13)

    axes[0, 0].imshow(ref_image, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].add_patch(Rectangle((sc1, r0), sc2 - sc1, r1 - r0, linewidth=2,
                                   edgecolor="#E24B4A", facecolor="none", label="Shift"))
    axes[0, 0].add_patch(Rectangle((nc1, r0), nc2 - nc1, r1 - r0, linewidth=2,
                                   edgecolor="#378ADD", facecolor="none", label="NoShift"))
    axes[0, 0].axhline(kc_abs, color="#3B6D11", ls=":", lw=1.5, label=f"KC row = {kc_abs}")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    axes[0, 0].set_title("Camera image — detected ROIs"); axes[0, 0].axis("off")

    axes[0, 1].imshow(var_map, cmap="hot")
    axes[0, 1].set_title("Pixel variance map\n(bright = changes between images)")
    axes[0, 1].axis("off")

    axes[1, 0].plot(row_var, lw=0.8)
    axes[1, 0].axvline(r0, color="#E24B4A", ls="--", label=f"ROW_START={r0}")
    axes[1, 0].axvline(r1, color="#E24B4A", ls="--", label=f"ROW_END={r1}")
    axes[1, 0].axvline(kc_abs, color="#3B6D11", ls=":", label=f"KC abs={kc_abs}")
    axes[1, 0].set_xlabel("Row (y)"); axes[1, 0].set_ylabel("mean variance / cols")
    axes[1, 0].set_title("Row variance -> ROW_START/END, KC"); axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(col_var, lw=0.8)
    for x, c, lab in [(sc1, "#E24B4A", "SHIFT_C1"), (sc2, "#E24B4A", "SHIFT_C2"),
                      (nc1, "#378ADD", "NOSHIFT_C1"), (nc2, "#378ADD", "NOSHIFT_C2")]:
        axes[1, 1].axvline(x, color=c, ls="--", label=f"{lab}={x}")
    axes[1, 1].set_xlabel("Column (x)"); axes[1, 1].set_ylabel("mean variance / rows")
    axes[1, 1].set_title("Column variance -> Shift & NoShift cols"); axes[1, 1].legend(fontsize=8)

    plt.tight_layout(); plt.savefig(output_path, dpi=130); plt.show()
    print(f"\n  Plot saved to: {output_path}")


if __name__ == "__main__":
    results, ref_img, row_var, col_var, var_map = detect_roi(IMAGE_PATHS)
    plot_results(results, ref_img, row_var, col_var, var_map, OUTPUT_PLOT)
    print("\n=== Copy these into your calibration script ===\n")
    for klab in ("CAM_ROW_START", "CAM_ROW_END", "CAM_NOSHIFT_C1", "CAM_NOSHIFT_C2",
                 "CAM_SHIFT_C1", "CAM_SHIFT_C2", "KC"):
        print(f"{klab:15s} = {results[klab]}")
