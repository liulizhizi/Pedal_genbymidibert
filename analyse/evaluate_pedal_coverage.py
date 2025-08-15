import pretty_midi
import numpy as np
import os
import json
import pandas as pd
from scipy.stats import entropy, pearsonr

# ============ Configuration ============
maestro_json = "../maestro_splits.json"  # Path to MAESTRO dataset split JSON
gt_root = "../"  # Root directory of ground truth MIDI files
pred_folder_name = "256_Full"  # Folder name of predictions (e.g., "256_Full" or "256_Partial")
output_dir = f"comparison_results_devklcorr_{pred_folder_name}"  # Output folder
time_step = 0.01  # Time resolution for pedal sampling (seconds)
outlier_std_threshold = 3  # Threshold (in std dev) to remove extreme values
results = []
# =======================================

os.makedirs(output_dir, exist_ok=True)
pred_root = os.path.join("../", pred_folder_name + "/output_midi")


# ==== Extract pedal curve from a MIDI file ====
def extract_pedal_curve(midi_path, max_time, time_step=0.01):
    """
    Extract continuous sustain pedal curve from a MIDI file.
    Returns a numpy array of normalized pedal values [0, 1] sampled at `time_step`.

    Parameters:
    ----------
    midi_path : str
        Path to MIDI file
    max_time : float
        Duration of the curve (seconds)
    time_step : float
        Sampling interval (seconds)
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI: {midi_path}, {e}")
        return np.zeros(int(max_time / time_step) + 1)

    # Collect all sustain pedal CC events (CC 64)
    pedal_cc = []
    for instr in midi.instruments:
        pedal_cc.extend([cc for cc in instr.control_changes if cc.number == 64])

    # Initialize pedal curve
    times = np.arange(0, max_time, time_step)
    pedal_values = np.zeros_like(times)

    if not pedal_cc:
        return pedal_values

    pedal_cc.sort(key=lambda x: x.time)

    # Fill pedal curve using CC values
    for i in range(len(pedal_cc) - 1):
        start_t, start_val = pedal_cc[i].time, pedal_cc[i].value
        end_t = pedal_cc[i + 1].time
        idx_start = max(0, min(int(start_t / time_step), len(times)))
        idx_end = max(0, min(int(end_t / time_step), len(times)))
        pedal_values[idx_start:idx_end] = start_val

    # Fill remaining time with last value
    last_t, last_val = pedal_cc[-1].time, pedal_cc[-1].value
    idx_last = max(0, min(int(last_t / time_step), len(times)))
    pedal_values[idx_last:] = last_val

    return pedal_values / 127.0  # Normalize to [0, 1]


# ==== Metrics ====
def deviation_multiple(pred_curve, gt_curve):
    """Compute deviation multiple: mean difference normalized by GT std."""
    std = np.std(gt_curve)
    std = std if std > 1e-6 else 1e-6
    return np.mean((pred_curve - gt_curve) / std)


def kl_divergence(pred_curve, gt_curve, bins=50):
    """Compute KL divergence between predicted and GT pedal distributions."""
    hist_p, _ = np.histogram(pred_curve, bins=bins, range=(0, 1), density=True)
    hist_q, _ = np.histogram(gt_curve, bins=bins, range=(0, 1), density=True)
    hist_p += 1e-8  # Avoid zero
    hist_q += 1e-8
    return entropy(hist_p, hist_q)


def pearson_corr(pred_curve, gt_curve):
    """Compute Pearson correlation coefficient between predicted and GT curves."""
    if np.std(pred_curve) < 1e-8 or np.std(gt_curve) < 1e-8:
        return 0.0
    r, _ = pearsonr(pred_curve, gt_curve)
    return r


# =========================
# Batch process test set
# =========================
with open(maestro_json, 'r') as f:
    data = json.load(f)

for test_file in data["test"]:
    gt_path = os.path.join(gt_root, test_file)
    filename = os.path.splitext(os.path.basename(gt_path))[0]
    pred_path = os.path.join(pred_root, f"{filename}_output.mid")

    if not os.path.exists(pred_path):
        print(f"❌ Missing prediction: {pred_path}")
        continue

    print(f"🔍 Comparing (pedal only): {filename}")

    try:
        gt_midi = pretty_midi.PrettyMIDI(gt_path)
        pred_midi = pretty_midi.PrettyMIDI(pred_path)
    except Exception as e:
        print(f"Error loading MIDI: {filename}, {e}")
        continue

    max_time = max(gt_midi.get_end_time(), pred_midi.get_end_time())
    gt_curve = extract_pedal_curve(gt_path, max_time, time_step)
    pred_curve = extract_pedal_curve(pred_path, max_time, time_step)

    dev_mult = deviation_multiple(pred_curve, gt_curve)
    kl = kl_divergence(pred_curve, gt_curve)
    corr = pearson_corr(pred_curve, gt_curve)

    results.append({
        "filename": filename,
        "deviation_multiple": dev_mult,
        "kl_divergence": kl,
        "pearson_r": corr
    })

# =========================
# Save raw CSV
# =========================
df = pd.DataFrame(results)
csv_path = os.path.join(output_dir, f"pedal_metrics_devklcorr_{pred_folder_name}.csv")
df.to_csv(csv_path, index=False)
print(f"\n✅ CSV saved: {csv_path}")

# =========================
# Compute mean ± std, remove outliers
# =========================
metrics = ['deviation_multiple', 'kl_divergence', 'pearson_r']

# Original mean ± std
mean_values = df[metrics].mean()
std_values = df[metrics].std()
print("\n===== Original mean ± std =====")
for col in metrics:
    print(f"{col}: {mean_values[col]:.2f} ± {std_values[col]:.2f}")

# Remove extreme outliers
df_no_outliers = df.copy()
initial_len = len(df)
for col in metrics:
    mean = df[col].mean()
    std = df[col].std()
    df_no_outliers = df_no_outliers[
        (df_no_outliers[col] >= mean - outlier_std_threshold * std) &
        (df_no_outliers[col] <= mean + outlier_std_threshold * std)
        ]

removed_count = initial_len - len(df_no_outliers)
mean_no_outliers = df_no_outliers[metrics].mean()
std_no_outliers = df_no_outliers[metrics].std()

print("\n===== Mean ± std after removing outliers =====")
for col in metrics:
    print(f"{col}: {mean_no_outliers[col]:.2f} ± {std_no_outliers[col]:.2f}")

print(f"\nRemoved {removed_count} extreme records (threshold: mean±{outlier_std_threshold}σ)")

# Save summary CSV
summary_df = pd.DataFrame({
    'Metric': metrics,
    'Mean': mean_values.values,
    'Std': std_values.values,
    'Mean_No_Outliers': mean_no_outliers.values,
    'Std_No_Outliers': std_no_outliers.values
})
summary_csv_path = os.path.join(output_dir, f"summary_{pred_folder_name}.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"\nResults saved to {summary_csv_path}")
