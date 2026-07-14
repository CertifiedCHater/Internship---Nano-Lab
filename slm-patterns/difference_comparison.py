import glob
import os
import re
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


FOLDER      = r""
GLOB        = "Capture_gray_*.bmp"
INTERACTIVE = True                       # True = click the box on a reference frame
BOX_ON      = None                       # file to draw the box on (None = middle frame)
MANUAL_BOX  = (0, 0, 0, 0)               # (x0,x1,y0,y1) when INTERACTIVE=False
BG_SIGMA    = 220                        # background smoothing; MUST exceed the square
REF_MARGIN  = 120                        # ring around the box excluded from the reference
OUT_SUBDIR  = "box_results"              # where the 255 figures are written


def load(p):
    return np.asarray(Image.open(p).convert("L"), dtype=float)


def parse_gray(name):
    n = re.findall(r"\d+", os.path.basename(name))
    return n[-1] if n else "?"


def phase_map(a):
    m = a > a.max() * 0.4
    ys, xs = np.where(m); r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    strip = a[r0:r1, (c0 + c1) // 2 - 100:(c0 + c1) // 2 + 100].mean(1); strip -= strip.mean()
    spec = np.abs(np.fft.rfft(strip * np.hanning(len(strip))))
    f = np.fft.rfftfreq(len(strip))[np.argmax(spec[3:]) + 3]
    Y = np.arange(a.shape[0])[:, None]
    demod = (a - gaussian_filter(a, 25)) * np.exp(-2j * np.pi * f * Y)
    lp = gaussian_filter(demod.real, 12) + 1j * gaussian_filter(demod.imag, 12)
    ph = np.angle(lp)
    bg = np.angle(gaussian_filter(lp.real, BG_SIGMA) + 1j * gaussian_filter(lp.imag, BG_SIGMA))
    resid = np.angle(np.exp(1j * (ph - bg)))
    beam = gaussian_filter(a, 15) > a.max() * 0.35
    return resid, beam


def measure(resid, beam, box, ref_margin=REF_MARGIN):
    x0, x1, y0, y1 = box
    inside = np.zeros(resid.shape, bool); inside[y0:y1, x0:x1] = True; inside &= beam
    grown = np.zeros(resid.shape, bool)
    grown[max(0, y0 - ref_margin):y1 + ref_margin, max(0, x0 - ref_margin):x1 + ref_margin] = True
    reference = beam & ~grown
    if not inside.any() or not reference.any():
        return np.nan
    step = np.angle(np.exp(1j * (np.median(resid[inside]) - np.median(resid[reference]))))
    return step / np.pi


def save_figure(a, resid, beam, box, step_pi, label, out_png):
    x0, x1, y0, y1 = box
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    title = f"gray {label}  —  phase step = {step_pi:+.3f} π" if np.isfinite(step_pi) \
            else f"gray {label}  —  phase step = n/a"
    fig.suptitle(title, fontweight="bold")
    ax[0].imshow(a, cmap="gray", vmin=0, vmax=255)
    ax[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, ec="lime", fc="none", lw=2))
    ax[0].set_title("raw + your region (green = shifted square)"); ax[0].axis("off")
    ax[1].imshow(np.where(beam, resid, np.nan), cmap="twilight", vmin=-1, vmax=1)
    ax[1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, ec="lime", fc="none", lw=2))
    ax[1].set_title("fringe-phase map (square stands out)"); ax[1].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def get_box_interactive(a):
    matplotlib.use("TkAgg")
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(a, cmap="gray", vmin=0, vmax=255)
    ax.set_title("CLICK the two opposite corners of the shifted square")
    pts = plt.ginput(2, timeout=0); plt.close(fig)
    matplotlib.use("Agg")
    (xa, ya), (xb, yb) = pts
    return (int(min(xa, xb)), int(max(xa, xb)), int(min(ya, yb)), int(max(ya, yb)))


def run(folder=FOLDER):
    files = sorted(glob.glob(os.path.join(folder, GLOB)))
    if not files:
        print("  no files matching", GLOB, "in", folder); return
    outdir = os.path.join(folder, OUT_SUBDIR); os.makedirs(outdir, exist_ok=True)
    print(f"  {len(files)} images -> {len(files)} figures in {outdir}")

    box = get_box_interactive(load(BOX_ON or files[len(files) // 2])) if INTERACTIVE else MANUAL_BOX
    print(f"  box (applied to all): x={box[0]}..{box[1]}, y={box[2]}..{box[3]}")

    with open(os.path.join(folder, "box_sweep.csv"), "w") as fh:
        fh.write("index,filename,gray,phase_step_pi\n")
        for i, f in enumerate(files):
            a = load(f)
            resid, beam = phase_map(a)
            step = measure(resid, beam, box)
            label = parse_gray(f)
            save_figure(a, resid, beam, box, step, label,
                        os.path.join(outdir, f"result_{i:04d}_gray{label}.png"))
            fh.write(f"{i},{os.path.basename(f)},{label},{step:.4f}\n")
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(files)}")
    print(f"  done: {len(files)} figures in {outdir}")


if __name__ == "__main__":
    run()
