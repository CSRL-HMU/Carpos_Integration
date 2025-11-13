import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# === SETTINGS ===
data_folder = "/home/csrl/PycharmProjects/CSRL/carpos/logs/"
out_folder = os.path.join(data_folder, "analysis")
os.makedirs(out_folder, exist_ok=True)

# weights and thresholds
w1, w2, w3 = 1, 1, 5
T_partial = 0.15 #0.3
T_strong = 0.35 #0.6

files = [f for f in os.listdir(data_folder) if f.endswith(".mat")]
print(f"Found {len(files)} .mat files in {data_folder}")

for fname in files:
    fpath = os.path.join(data_folder, fname)
    data = loadmat(fpath)

    # --- extract values ---
    try:
        palm_conf = float(np.ravel(data["palm_conf"]))
        tomato_conf = float(np.ravel(data["tomato_conf"]))
        iou = float(np.ravel(data["iou"]))
    except KeyError:
        print(f"Skipping {fname} (missing fields)")
        continue

    # --- compute occlusion metric ---
    M = (w1 * (1 - palm_conf) + w2 * (1 - tomato_conf) + w3 * iou) / (w1 + w2 + w3)

    # --- fallback for missing detections ---
    if (iou == 0 and tomato_conf < 0.1) or (palm_conf < 0.1 and tomato_conf < 0.1):
        M = 1.0

    # --- classify occlusion ---
    if M <= T_partial:
        level = "none"
    elif M <= T_strong:
        level = "partial"
    else:
        level = "strong"

    print(f"{fname}: M={M:.3f} → {level.upper()} occlusion")

    # --- save simple plot ---
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar([0], [M], color="black")
    ax.axhline(T_partial, color="gold", linestyle="--", label="Partial threshold")
    ax.axhline(T_strong, color="red", linestyle="--", label="Strong threshold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Occlusion metric M")
    ax.set_title(f"{fname}\n{level.upper()} occlusion")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    out_path = os.path.join(out_folder, fname.replace(".mat", "_metric.png"))
    plt.savefig(out_path)
    plt.close(fig)

print(f"\nAnalysis complete. Plots saved in {out_folder}")
