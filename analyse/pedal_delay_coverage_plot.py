import os
import json
import pretty_midi
import matplotlib.pyplot as plt
import numpy as np

# ================= Configuration =================
CONFIG = {
    "version": "partial",  # "partial" or "full"

    # MAESTRO split JSON
    "maestro_json": "../maestro_splits.json",
    "midi_root": "../",

    # Prediction MIDI root directories
    "pred_root_partial": "../256_Partial/output_midi",
    "pred_root_full": "../256_Full/output_midi",

    # Plot output directories
    "plot_dir_partial": "../delay_coverage_partial/",
    "plot_dir_full": "../delay_coverage_full/",
}

# Choose path according to version
def select_path(version, key_partial, key_full):
    return CONFIG[key_partial] if version == "partial" else CONFIG[key_full]

# ================= Helper Functions =================
# Get predicted MIDI path corresponding to a ground truth MIDI
def get_pred_path(gt_path, pred_root):
    filename = os.path.splitext(os.path.basename(gt_path))[0]
    return os.path.join(pred_root, f"{filename}_output.mid")

# Plot heatmap showing note coverage by pedal segments
def plot_coverage_heatmap(data, title, filename):
    if not data:
        print(f"No data to plot: {title}")
        return
    times = np.array([t for _, t in data])
    pitches = np.array([p for p, _ in data])
    heatmap, xedges, yedges = np.histogram2d(
        times, pitches, bins=[200, 60], range=[[0, max(times)], [20, 108]]
    )
    plt.figure(figsize=(12, 6))
    plt.imshow(heatmap.T, origin='lower', aspect='auto', cmap='hot',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar(label='Pedal Coverage Count')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("MIDI Pitch")
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.show()

# ================= Main Function =================
def main():
    version = CONFIG["version"]
    pred_root = select_path(version, "pred_root_partial", "pred_root_full")
    plot_dir = select_path(version, "plot_dir_partial", "plot_dir_full")
    os.makedirs(plot_dir, exist_ok=True)

    # Load MAESTRO test split
    with open(CONFIG["maestro_json"], 'r') as f:
        data = json.load(f)

    # Lists to collect metrics
    release_delays_gt = []
    release_delays_pred = []
    coverage_gt = []
    coverage_pred = []

    # Process each test MIDI file
    for rel_path in data["test"]:
        gt_path = os.path.join(CONFIG["midi_root"], rel_path)
        pred_path = get_pred_path(gt_path, pred_root)

        # Load ground truth and predicted MIDI
        try:
            gt_midi = pretty_midi.PrettyMIDI(gt_path)
            pred_midi = pretty_midi.PrettyMIDI(pred_path)
        except Exception as e:
            print(f"Failed to load: {rel_path} — {e}")
            continue

        # Extract pedal segments and compute metrics for GT and prediction
        for name, midi_obj, delays, coverage in [
            ("GT", gt_midi, release_delays_gt, coverage_gt),
            ("Pred", pred_midi, release_delays_pred, coverage_pred)
        ]:
            if not midi_obj.instruments:
                continue
            instr = midi_obj.instruments[0]
            notes = instr.notes
            cc_events = [cc for cc in instr.control_changes if cc.number == 64]

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

            # Compute release delays and note coverage
            for start, end in pedal_segments:
                covered_notes = [n for n in notes if n.start >= start and n.start < end]
                if not covered_notes:
                    continue
                last_note_end = max(n.end for n in covered_notes)
                delay = end - last_note_end
                delays.append(delay)
                for n in covered_notes:
                    coverage.append((n.pitch, n.start))

    # ===== Plot pedal release delay distributions =====
    plt.figure(figsize=(10, 4))
    plt.hist(np.array(release_delays_gt)*1000, bins=100, range=(-4000,1000),
             alpha=0.6, label="Ground Truth", color="green", edgecolor="black")
    plt.hist(np.array(release_delays_pred)*1000, bins=100, range=(-4000,1000),
             alpha=0.6, label="Prediction", color="orange", edgecolor="black")
    plt.xlim(-4000, 1000)
    plt.title("Pedal Release Delay Distribution Comparison")
    plt.xlabel("Delay after last note (ms)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"release_delay_comparison_{version}.png"), dpi=300)
    plt.show()

    # ===== Plot pedal coverage heatmaps =====
    plot_coverage_heatmap(coverage_gt, f"Ground Truth Pedal Coverage Heatmap ({version})",
                          os.path.join(plot_dir, f"pedal_coverage_gt_{version}.png"))
    plot_coverage_heatmap(coverage_pred, f"Predicted Pedal Coverage Heatmap ({version})",
                          os.path.join(plot_dir, f"pedal_coverage_pred_{version}.png"))

if __name__ == "__main__":
    main()
