import pretty_midi
import numpy as np
import os
import json
import pandas as pd
from scipy.stats import entropy, pearsonr

# ===== Configuration =====
maestro_json = "../maestro_splits.json"  # Path to MAESTRO dataset split JSON
gt_root = "../"  # Root directory of ground truth MIDI files
pred_folder_name = "256_Full"  # Folder name of predicted MIDI outputs 256_Full or 256_Partial
output_dir = f"comparison_results_onset_{pred_folder_name}"  # Output directory
time_step = 0.01  # Time resolution for onset curves (seconds)
outlier_std_threshold = 3  # Threshold (in std deviations) to remove extreme values

os.makedirs(output_dir, exist_ok=True)
pred_root = os.path.join("../", pred_folder_name + "/output_midi")
results = []


# ===== Function definitions =====
def extract_pedal_onsets(midi_path):
    """
    Extract sustain pedal onset times from a MIDI file.
    Onset is defined as the moment CC64 value crosses >=64 from below.

    Parameters
    ----------
    midi_path : str
        Path to the MIDI file.

    Returns
    -------
    np.ndarray
        Array of onset times (seconds).
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI: {midi_path}, {e}")
        return np.array([])

    onsets = []
    for instr in midi.instruments:
        cc64 = [cc for cc in instr.control_changes if cc.number == 64]
        cc64.sort(key=lambda x: x.time)
        for i, cc in enumerate(cc64):
            if cc.value >= 64:
                if i == 0 or cc64[i - 1].value < 64:
                    onsets.append(cc.time)
    return np.array(onsets)


def deviation_multiple_onset(pred_onsets, gt_onsets):
    """
    Compute deviation multiple between predicted and ground truth onset times.
    Uses mean difference normalized by standard deviation of GT onsets.
    """
    if len(gt_onsets) == 0 or len(pred_onsets) == 0:
        return np.nan
    min_len = min(len(pred_onsets), len(gt_onsets))
    std = np.std(gt_onsets) if np.std(gt_onsets) > 1e-6 else 1e-6
    return np.mean((pred_onsets[:min_len] - gt_onsets[:min_len]) / std)


def kl_divergence_onset(pred_onsets, gt_onsets, max_time, bins=50):
    """
    Compute KL divergence between predicted and GT onset distributions.
    Converts onset times into histograms.
    """
    hist_p, _ = np.histogram(pred_onsets, bins=bins, range=(0, max_time), density=True)
    hist_q, _ = np.histogram(gt_onsets, bins=bins, range=(0, max_time), density=True)
    hist_p += 1e-8  # avoid zero
    hist_q += 1e-8
    return entropy(hist_p, hist_q)


def pearson_corr_onset(pred_onsets, gt_onsets, max_time, time_step=0.01):
    """
    Compute Pearson correlation by converting onset times into binary time series.
    """
    times = np.arange(0, max_time + time_step, time_step)
    gt_curve = np.zeros_like(times)
    pred_curve = np.zeros_like(times)
    gt_idx = (gt_onsets / time_step).astype(int)
    pred_idx = (pred_onsets / time_step).astype(int)
    gt_curve[gt_idx] = 1
    pred_curve[pred_idx] = 1
    if np.std(gt_curve) < 1e-8 or np.std(pred_curve) < 1e-8:
        return 0.0
    r, _ = pearsonr(pred_curve, gt_curve)
    return r


# ===== Batch process test set =====
with open(maestro_json, 'r') as f:
    data = json.load(f)

for test_file in data["test"]:
    gt_path = os.path.join(gt_root, test_file)
    filename = os.path.splitext(os.path.basename(gt_path))[0]
    pred_path = os.path.join(pred_root, f"{filename}_output.mid")
    if not os.path.exists(pred_path):
        print(f"❌ Missing prediction: {pred_path}")
        continue

    gt_onsets = extract_pedal_onsets(gt_path)
    pred_onsets = extract_pedal_onsets(pred_path)

    try:
        gt_midi = pretty_midi.PrettyMIDI(gt_path)
        pred_midi = pretty_midi.PrettyMIDI(pred_path)
    except Exception as e:
        print(f"Error loading MIDI: {filename}, {e}")
        continue
    max_time = max(gt_midi.get_end_time(), pred_midi.get_end_time())

    # Compute metrics
    dev_mult = deviation_multiple_onset(pred_onsets, gt_onsets)
    kl = kl_divergence_onset(pred_onsets, gt_onsets, max_time)
    corr = pearson_corr_onset(pred_onsets, gt_onsets, max_time, time_step)

    results.append({
        "filename": filename,
        "deviation_multiple": dev_mult,
        "kl_divergence": kl,
        "pearson_r": corr
    })

# ===== Save CSV =====
df = pd.DataFrame(results)
csv_path = os.path.join(output_dir, f"pedal_onset_metrics_{pred_folder_name}.csv")
df.to_csv(csv_path, index=False)
print(f"\n✅ CSV saved: {csv_path}")

# ===== Compute mean ± std =====
metrics = ['deviation_multiple', 'kl_divergence', 'pearson_r']
mean_values = df[metrics].mean()
std_values = df[metrics].std()

# ===== Remove outliers and recompute mean ± std =====
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

# ===== Print results =====
print("\n===== Original mean ± std =====")
for col in metrics:
    print(f"{col}: {mean_values[col]:.2f} ± {std_values[col]:.2f}")

print("\n===== Mean ± std after removing outliers =====")
for col in metrics:
    print(f"{col}: {mean_no_outliers[col]:.2f} ± {std_no_outliers[col]:.2f}")

print(f"\nRemoved {removed_count} extreme records (threshold: mean ± {outlier_std_threshold}σ)")

# ===== Save summary.csv =====
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
