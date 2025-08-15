import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from matplotlib.ticker import MaxNLocator

# ================= Configuration =================
CONFIG = {
    "version": "partial",  # "partial" or "full"

    "maestro_json": "../maestro_splits.json",
    "gt_root": "../",

    # Prediction MIDI roots
    "pred_root_partial": "../256_Partial/output_midi",
    "pred_root_full": "../256_Full/output_midi",

    # Output directories for results
    "output_dir_partial": "heatmap_partial",
    "output_dir_full": "heatmap_full",

    # Heatmap parameters
    "time_bins": 200,
    "pitch_range": (20, 108)
}

# Select configuration path according to version
def select_path(version, key_partial, key_full):
    return CONFIG[key_partial] if version == "partial" else CONFIG[key_full]

# ================= Helper Functions =================
# Extract pedal coverage information: notes covered by pedal segments
def extract_pedal_coverage(midi_path):
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI: {midi_path}, {e}")
        return [], 0.0

    if not midi.instruments:
        return [], 0.0
    instr = midi.instruments[0]
    notes = instr.notes
    cc_events = [cc for cc in instr.control_changes if cc.number == 64]

    # Identify pedal segments where sustain pedal is pressed
    pedal_segments = []
    i = 0
    while i < len(cc_events):
        if cc_events[i].value >= 64:
            start = cc_events[i].time
            for j in range(i+1, len(cc_events)):
                if cc_events[j].value < 64:
                    end = cc_events[j].time
                    pedal_segments.append((start, end))
                    i = j
                    break
        i += 1

    # Collect (pitch, start_time) of notes covered by pedal
    coverage = []
    for start, end in pedal_segments:
        for n in notes:
            if start <= n.start < end:
                coverage.append((n.pitch, n.start))
    max_time = midi.get_end_time()
    return coverage, max_time

# Build 2D heatmap of pedal coverage over time and pitch
def build_heatmap(coverage, max_time):
    if not coverage:
        return np.zeros((CONFIG["time_bins"], CONFIG["pitch_range"][1] - CONFIG["pitch_range"][0]))
    times = np.array([t for _, t in coverage])
    pitches = np.array([p for p, _ in coverage])
    heatmap, _, _ = np.histogram2d(
        times, pitches,
        bins=[CONFIG["time_bins"], CONFIG["pitch_range"][1] - CONFIG["pitch_range"][0]],
        range=[[0, max_time], [CONFIG["pitch_range"][0], CONFIG["pitch_range"][1]]]
    )
    return heatmap

# Compare ground truth and predicted heatmaps and compute precision, recall, F1
def compare_heatmaps(gt_path, pred_path, save_prefix):
    gt_coverage, gt_max_time = extract_pedal_coverage(gt_path)
    pred_coverage, pred_max_time = extract_pedal_coverage(pred_path)
    max_time = max(gt_max_time, pred_max_time)

    heatmap_gt = build_heatmap(gt_coverage, max_time)
    heatmap_pred = build_heatmap(pred_coverage, max_time)

    # ==== Side-by-side heatmaps ====
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(heatmap_gt.T, origin='lower', aspect='auto', cmap='hot')
    axes[0].set_title("Ground Truth Pedal Coverage")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("MIDI Pitch")
    axes[1].imshow(heatmap_pred.T, origin='lower', aspect='auto', cmap='hot')
    axes[1].set_title("Predicted Pedal Coverage")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("MIDI Pitch")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_GT_Pred.png", dpi=300)
    plt.close()

    # ==== Difference heatmap ====
    diff = heatmap_pred - heatmap_gt
    vmax = np.max(np.abs(diff))
    plt.figure(figsize=(12, 6))
    plt.imshow(diff.T, origin='lower', aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax)
    plt.title("Pedal Coverage Difference (Pred - GT)")
    plt.xlabel("Time")
    plt.ylabel("MIDI Pitch")
    plt.colorbar(label="Difference")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_Difference.png", dpi=300)
    plt.close()

    # ==== Compute precision, recall, F1 ====
    gt_mask = heatmap_gt > 0
    pred_mask = heatmap_pred > 0
    TP = np.sum(gt_mask & pred_mask)
    FP = np.sum(~gt_mask & pred_mask)
    FN = np.sum(gt_mask & ~pred_mask)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1

# ================= Main Function =================
def main():
    version = CONFIG["version"]
    pred_root = select_path(version, "pred_root_partial", "pred_root_full")
    output_dir = select_path(version, "output_dir_partial", "output_dir_full")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    with open(CONFIG["maestro_json"], 'r') as f:
        data = json.load(f)

    # Process each test MIDI file
    for test_file in data["test"]:
        gt_path = os.path.join(CONFIG["gt_root"], test_file)
        filename = os.path.splitext(os.path.basename(gt_path))[0]
        pred_path = os.path.join(pred_root, f"{filename}_output.mid")
        save_prefix = os.path.join(output_dir, filename)

        if not os.path.exists(pred_path):
            print(f"❌ Missing prediction: {pred_path}")
            continue

        print(f"🔍 Comparing: {filename}")
        precision, recall, f1 = compare_heatmaps(gt_path, pred_path, save_prefix)
        print(f"  → Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        results.append({"filename": filename, "precision": precision, "recall": recall, "f1": f1})

    # Save metrics to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "pedal_coverage_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ CSV saved: {csv_path}")

    # Plot histogram of F1 scores
    if not df.empty:
        mean_f1 = df["f1"].mean()
        plt.figure(figsize=(10, 6))
        plt.hist(df["f1"], bins=20, color='skyblue', edgecolor='black')
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.axvline(mean_f1, color='red', linestyle='dashed', linewidth=2, label=f'Mean F1 = {mean_f1:.4f}')
        plt.xlabel("F1 Score")
        plt.ylabel("Number of Files")
        plt.title(f"F1 Score Distribution of Pedal Coverage ({version})")
        plt.legend()
        plt.tight_layout()
        hist_path = os.path.join(output_dir, "F1_Distribution.png")
        plt.savefig(hist_path, dpi=300)
        plt.show()
        print(f"📊 F1 histogram saved: {hist_path}")
    else:
        print("⚠️ No data to plot.")

if __name__ == "__main__":
    main()
