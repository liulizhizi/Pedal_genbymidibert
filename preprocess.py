from miditok import REMI
import json
import pandas as pd
from collections import defaultdict
import numpy as np
import os

# Load tokenizer configuration and initialize
tokenizer_path = "./tokenizer_hi2.json"
tokenizer = REMI(params=tokenizer_path)

# Load train/validation/test splits
with open('maestro_splits.json', 'r') as f:
    data = json.load(f)

train_files = data["train"]
val_files = data["validation"]
test_files = data["test"]


def extract_fields_from_tok_sequence(tok_sequence):
    """
    Extracts fields from a tokenized MIDI sequence.

    Args:
        tok_sequence: Tokenized MIDI sequence (from miditok REMI)

    Returns:
        fields: dict containing tokens split by feature and metadata
    """
    fields = defaultdict(list)
    for token in tok_sequence[0].tokens:
        if "_" in token:
            field, value = token.split("_", 1)
            fields[field].append(value)
        else:
            fields[token].append(None)
    fields.update({
        "ids": tok_sequence[0].ids,
        "bytes": tok_sequence[0].bytes,
        "events": tok_sequence[0].events,
        "are_ids_encoded": tok_sequence[0].are_ids_encoded,
        "_ticks_bars": tok_sequence[0]._ticks_bars,
        "_ticks_beats": tok_sequence[0]._ticks_beats,
        "_ids_decoded": tok_sequence[0]._ids_decoded
    })
    return fields


def build_token_id_dict(tok_sequence, fields):
    """Creates a list of tuples (token, token_id)."""
    return list(zip(tok_sequence[0].tokens, fields["ids"]))


def process_keys(pairs):
    """Extract the base feature name from token keys."""
    return [(key.split('_')[0], value) for key, value in pairs]


def process_pairs(pairs):
    """
    Convert a list of (key, value) pairs into a structured DataFrame.

    Returns:
        df: pandas DataFrame with columns ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']
    """
    fixed_columns = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']
    rows = []
    current_row = {}

    for key, value in pairs:
        current_row[key] = value
        if key == 'Duration':
            # Ensure all fixed columns exist
            for col in fixed_columns:
                if col not in current_row:
                    current_row[col] = None
            filtered_row = {k: v for k, v in current_row.items() if k in fixed_columns}
            rows.append(filtered_row)
            current_row = {}

    return pd.DataFrame(rows, columns=fixed_columns)


def preprocess_dataframe(df):
    """
    Basic preprocessing: reset indices, fill missing values, and convert types.

    - Re-index Bar==4 rows sequentially.
    - Forward-fill Position values.
    """
    df = df.reset_index(drop=True)
    df['Bar'] = df['Bar'].fillna(0).astype(int)

    # Number rows where Bar == 4 sequentially
    counter = 1
    for i in range(len(df)):
        if df.at[i, 'Bar'] == 4:
            df.at[i, 'Bar'] = counter
            counter += 1

    df['Position'] = df['Position'].ffill().astype('int32')
    df['Pitch'] = df['Pitch'].fillna(0).astype(int)
    df['Velocity'] = df['Velocity'].fillna(0).astype(int)
    df['Duration'] = df['Duration'].fillna(0).astype(int)
    df['Pedal'] = df['Pedal'].fillna(0).astype(int)
    return df


def process_bar_column(df):
    """
    Process the 'Bar' column:
    - Insert new rows for segment headers and zero rows.
    - Reset current Bar to 0 to indicate intra-segment events.
    """
    new_df = pd.DataFrame(columns=df.columns)

    for _, row in df.iterrows():
        current_bar = row['Bar']

        if pd.notna(current_bar) and current_bar != 0:
            # Create header row
            data_row = pd.Series(0, index=df.columns)
            data_row['Bar'] = current_bar

            # Create zero row
            zero_row = pd.Series(0, index=df.columns)

            # Append both rows
            temp_df = pd.DataFrame([data_row, zero_row])
            new_df = pd.concat([new_df, temp_df], ignore_index=True)

            # Reset current row's Bar to 0
            row = row.copy()
            row['Bar'] = 0

        new_df = pd.concat([new_df, row.to_frame().T], ignore_index=True)

    return new_df


def process_pedal_special_values(df, max_len=10):
    """
    Handle special Pedal values (2882 for press events) and split long segments into subsegments.

    - Inserts two empty rows at the beginning of subsegments (after the first).
    - Assign Pedal_1 / Position_1 / Duration_1 for each subsegment.

    Args:
        df: input DataFrame with ['Bar','Position','Pitch','Velocity','Duration','Pedal']
        max_len: maximum non-event rows per subsegment

    Returns:
        df: processed DataFrame ready for feature extraction
    """
    df = df.copy()
    df.fillna(0, inplace=True)

    # Initialize new Pedal columns
    df['Pedal_1'] = df['Pedal'].where(df['Pedal'] == 2882, 0)
    df['Position_1'] = df['Position'].where(df['Pedal'] == 2882, 0)
    df['Duration_1'] = df['Duration'].where(df['Pedal'] == 2882, 0)

    # Mark segments: Bar != 0 indicates new segment
    df['seg_id'] = df['Bar'].ne(0).cumsum()
    df = df.reset_index(drop=True)
    df['row_idx'] = df.index

    all_result = []

    for seg_id, seg_df in df.groupby('seg_id'):
        seg_df = seg_df.reset_index(drop=True)
        seg_df['local_idx'] = seg_df.index

        # Split into event and non-event rows
        events = seg_df[seg_df['Pedal'].isin([2882])].copy()
        contents = seg_df[seg_df['Pedal'] == 0].copy().reset_index(drop=True)
        contents['content_idx'] = contents.index

        # Map local_idx to content_idx
        local_to_content = contents.set_index('local_idx')['content_idx'].to_dict()

        # Split contents into subsegments
        num_chunks = (len(contents) // max_len) + 1
        subseg_list = []

        for i in range(num_chunks):
            start = i * max_len
            end = start + max_len
            chunk = contents.iloc[start:end].copy()

            if i > 0:
                blank = pd.DataFrame(0, index=range(2), columns=chunk.columns)
                blank['Bar'] = 0
                chunk = pd.concat([blank, chunk], ignore_index=True)

            chunk['subseg_id'] = f"{seg_id}_{i}"
            subseg_list.append(chunk)

        # Assign events to subsegments
        def assign_subseg_id(row):
            cond = (contents['local_idx'] <= row['local_idx'])
            if not cond.any():
                return f"{seg_id}_0"
            nearest_local = contents.loc[cond, 'local_idx'].max()
            nearest_content_idx = local_to_content.get(nearest_local, 0)
            chunk_id = nearest_content_idx // max_len
            return f"{seg_id}_{int(chunk_id)}"

        events['subseg_id'] = events.apply(assign_subseg_id, axis=1)
        subseg_list.append(events)

        merged = pd.concat(subseg_list, ignore_index=True)
        all_result.append(merged)

    merged_all = pd.concat(all_result, ignore_index=True)
    result = merged_all[merged_all['Pedal'] == 0].copy()

    # Fill Pedal_1 / Position_1 / Duration_1
    for subseg_id, group in result.groupby('subseg_id'):
        orig = merged_all[merged_all['subseg_id'] == subseg_id]
        p1_list = orig[orig['Pedal_1'] != 0][['Pedal_1', 'Position_1', 'Duration_1']].values.tolist()

        idxs = group.index.tolist()
        for i, idx in enumerate(idxs):
            if i < len(p1_list):
                result.loc[idx, ['Pedal_1', 'Position_1', 'Duration_1']] = p1_list[i]
            else:
                result.loc[idx, ['Pedal_1', 'Position_1', 'Duration_1']] = [0, 0, 0]

    # Drop temporary columns
    result = result.drop(columns=['Pedal', 'subseg_id', 'seg_id', 'row_idx', 'local_idx', 'content_idx'],
                         errors='ignore')
    return result.reset_index(drop=True)


def df_to_feature_vector_11dim(df):
    """Convert DataFrame to 11-dimensional feature matrix (channels x sequence length)."""
    cols_to_fill = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration']
    for col in cols_to_fill:
        df[col] = df[col].fillna(0).astype(int)

    return df.values.T  # (channels, sequence length)


def process_sequences_with_sliding_window(input_data, max_length=256, stride=224):
    """
    Apply sliding window slicing to sequences and generate attention masks.

    Args:
        input_data: list of np.ndarray, each (channels, seq_len)
        max_length: maximum window length
        stride: sliding step size (< max_length ensures overlap)

    Returns:
        processed_data: np.ndarray (num_windows, max_length, channels)
        processed_masks: np.ndarray (num_windows, max_length)
    """
    processed_chunks = []
    processed_masks = []

    for arr in input_data:
        arr = arr.astype(np.int32)
        channels, total_len = arr.shape

        start_idx = 0
        while start_idx < total_len:
            end_idx = start_idx + max_length
            chunk = arr[:, start_idx:end_idx]
            chunk_len = chunk.shape[1]

            mask = np.ones(max_length, dtype=np.int32)

            if chunk_len < max_length:
                pad_width = ((0, 0), (0, max_length - chunk_len))
                chunk = np.pad(chunk, pad_width, mode='constant', constant_values=0)
                mask[chunk_len:] = 0

            chunk = chunk.T  # (max_length, channels)
            processed_chunks.append(chunk)
            processed_masks.append(mask)

            start_idx += stride

    return np.stack(processed_chunks, axis=0), np.stack(processed_masks, axis=0)


def process_dataset(files, save_dir, dataset_type):
    """General dataset processing pipeline for MIDI files."""
    feature_matrices = []
    processed_files = []

    for midi_path in files:
        tokens = tokenizer(midi_path)
        fields = extract_fields_from_tok_sequence(tokens)
        token_id_dict = build_token_id_dict(tokens, fields)
        processed_pairs = process_keys(token_id_dict)

        df = process_pairs(processed_pairs)

        # Pedal and bar processing
        df = process_bar_column(df)
        df = preprocess_dataframe(df)
        df = process_pedal_special_values(df)

        feature_matrix = df_to_feature_vector_11dim(df)
        feature_matrices.append(feature_matrix)
        processed_files.append(midi_path)

    if not feature_matrices:
        print(f"{dataset_type} dataset: no valid data to save")
        return

    processed_data, processed_masks = process_sequences_with_sliding_window(feature_matrices)

    # Save paths
    data_filename = f"{dataset_type}_data.npy"
    mask_filename = f"{dataset_type}_mask.npy"
    save_data_path = os.path.join(save_dir, data_filename)
    save_mask_path = os.path.join(save_dir, mask_filename)

    try:
        np.save(save_data_path, processed_data)
        print(f"\n{dataset_type} data saved successfully at {save_data_path}")
        print(f"Data shape: {processed_data.shape}")

        np.save(save_mask_path, processed_masks)
        print(f"{dataset_type} mask saved successfully at {save_mask_path}")
        print(f"Mask shape: {processed_masks.shape}")

        # Save metadata
        metadata = {
            "file_count": len(feature_matrices),
            "total_samples": processed_data.shape[0],
            "mask_samples": processed_masks.shape[0],
            "max_length": 256,
            "feature_dims": 6,
            "processed_files": processed_files
        }

        meta_path = os.path.join(save_dir, f"{dataset_type}_data.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        print(f"{dataset_type} data saving failed: {str(e)}")


if __name__ == "__main__":
    save_dir = "processed_data"
    os.makedirs(save_dir, exist_ok=True)

    process_dataset(train_files, save_dir, "train")
    process_dataset(val_files, save_dir, "val")
    process_dataset(test_files, save_dir, "test")
