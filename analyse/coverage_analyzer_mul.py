import os
import json
import pretty_midi
import pandas as pd
from glob import glob
from tqdm import tqdm

# ================= Configuration =================
CONFIG = {
    # Version of the predictions: "partial" or "mul"
    "version": "partial",

    # Path to MAESTRO JSON file with dataset splits
    "maestro_json": "../maestro_splits.json",

    # Root directory of ground truth MIDI files
    "gt_root": "../",

    # Root directories of predicted MIDI files for different versions
    "pred_root_partial": "../256/output_midi_small_deeper",
    "pred_root_mul": "../256mul/output_midi_small_deeper",

    # Output directories for per-file results
    "results_partial": "results/",
    "results_mul": "results_mul/",

    # Output directories for summary results
    "summary_partial": "summary/",
    "summary_mul": "summary_mul/",
}

# Select appropriate path based on the version
def select_path(version, key_partial, key_mul):
    return CONFIG[key_partial] if version == "partial" else CONFIG[key_mul]

# ================= MIDI Processing =================
def load_maestro_split(json_path):
    """Load MAESTRO dataset split JSON and return the test file list."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['test']

def get_pred_path(gt_path, pred_root):
    """Construct the predicted MIDI file path corresponding to a ground truth MIDI file."""
    filename = os.path.splitext(os.path.basename(gt_path))[0]
    return os.path.join(pred_root, f"{filename}_output.mid")

def extract_note_and_pedal(midi_path):
    """
    Extract note events and sustain pedal intervals from a MIDI file.

    Returns:
    --------
    note_df : pd.DataFrame
        Columns: note_start, note_end, pitch
    pedal_intervals : list of tuples
        Each tuple: (pedal_press_time, pedal_release_time)
    """
    midi = pretty_midi.PrettyMIDI(midi_path)

    # Extract note events from all non-drum instruments
    notes = []
    for instr in midi.instruments:
        if not instr.is_drum:
            for note in instr.notes:
                notes.append({'note_start': note.start, 'note_end': note.end, 'pitch': note.pitch})
    note_df = pd.DataFrame(notes)

    # Extract sustain pedal events (CC 64)
    pedals = []
    for instr in midi.instruments:
        for cc in instr.control_changes:
            if cc.number == 64:
                pedals.append((cc.time, cc.value))

    # Convert CC events to pressed intervals
    pedal_intervals = []
    is_pressed = False
    press_time = 0
    for time, value in pedals:
        if value >= 64 and not is_pressed:
            press_time = time
            is_pressed = True
        elif value < 64 and is_pressed:
            pedal_intervals.append((press_time, time))
            is_pressed = False

    return note_df, pedal_intervals

def get_mask(note_df, pedal_intervals):
    """
    Generate a binary mask indicating whether each note is covered by a sustain pedal.

    Returns:
    --------
    pd.Series: 1 if the note is under pedal, 0 otherwise
    """
    def is_covered(row):
        for pedal_start, pedal_end in pedal_intervals:
            if pedal_end > row['note_start'] and pedal_start < row['note_end']:
                return 1
        return 0

    return note_df.apply(is_covered, axis=1)

def compare_masks(note_df, mask_gt, mask_pred):
    """
    Compare ground truth and predicted pedal masks and label each note.

    Labels:
    - 'Both': GT and Pred both have pedal
    - 'OnlyGT': Only GT has pedal
    - 'OnlyPred': Only Pred has pedal
    - 'NoPedal': Neither has pedal
    """
    note_df = note_df.copy()
    note_df['mask_GT'] = mask_gt
    note_df['mask_Pred'] = mask_pred

    def classify(row):
        if row['mask_GT'] == 1 and row['mask_Pred'] == 1:
            return 'Both'
        elif row['mask_GT'] == 1 and row['mask_Pred'] == 0:
            return 'OnlyGT'
        elif row['mask_GT'] == 0 and row['mask_Pred'] == 1:
            return 'OnlyPred'
        else:
            return 'NoPedal'

    note_df['label'] = note_df.apply(classify, axis=1)
    return note_df

def process_pair(gt_path, pred_path):
    """
    Process a ground truth and predicted MIDI pair to generate comparison DataFrame.

    Returns:
    --------
    pd.DataFrame with note events, GT/Pred masks, and label classification.
    """
    note_df_gt, pedal_gt = extract_note_and_pedal(gt_path)
    _, pedal_pred = extract_note_and_pedal(pred_path)
    mask_gt = get_mask(note_df_gt, pedal_gt)
    mask_pred = get_mask(note_df_gt, pedal_pred)
    return compare_masks(note_df_gt, mask_gt, mask_pred)

# ================= Summary Analysis =================
def load_all_analysis_files(input_dir):
    """
    Load all CSV/XLSX analysis files in a directory and concatenate into a single DataFrame.
    Adds a column 'source_file' indicating the file origin.
    """
    all_files = glob(os.path.join(input_dir, "*.csv")) + glob(os.path.join(input_dir, "*.xlsx"))
    dfs = []
    for file in all_files:
        try:
            if file.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            df["source_file"] = os.path.basename(file)
            dfs.append(df)
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")
    return pd.concat(dfs, ignore_index=True)

# ================= Main Function =================
def main():
    """Main workflow for comparing predicted and ground truth MIDI files."""
    version = CONFIG["version"]
    pred_root = select_path(version, "pred_root_partial", "pred_root_mul")
    results_dir = select_path(version, "results_partial", "results_mul")
    summary_dir = select_path(version, "summary_partial", "summary_mul")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    # Load list of test MIDI files
    test_files = load_maestro_split(CONFIG["maestro_json"])

    # Process each GT-Pred MIDI pair
    for gt_path in tqdm(test_files, desc=f"Processing MIDI pairs ({version})"):
        gt_full_path = os.path.join(CONFIG["gt_root"], gt_path)
        pred_path = get_pred_path(gt_path, pred_root)
        if not os.path.exists(pred_path):
            print(f"Prediction MIDI not found for: {gt_path}")
            continue
        try:
            result_df = process_pair(gt_full_path, pred_path)
            filename = os.path.splitext(os.path.basename(gt_path))[0]
            result_df.to_csv(os.path.join(results_dir, f"{filename}_analysis.csv"), index=False)
        except Exception as e:
            print(f"Error processing {gt_path}: {e}")

    # Aggregate results across all files
    all_df = load_all_analysis_files(results_dir)
    confusion = pd.crosstab(all_df['mask_GT'], all_df['mask_Pred'], rownames=["GT"], colnames=["Pred"])
    label_counts = all_df['label'].value_counts(normalize=True).rename("proportion").reset_index()
    label_counts.columns = ['label', 'proportion']

    # Save summary results
    all_df.to_csv(os.path.join(summary_dir, "all_results_combined.csv"), index=False)
    confusion.to_csv(os.path.join(summary_dir, "confusion_matrix.csv"))
    label_counts.to_csv(os.path.join(summary_dir, "label_distribution.csv"), index=False)

    print(f"✅ Done. Version: {version}, files saved in {summary_dir}")

if __name__ == "__main__":
    main()
