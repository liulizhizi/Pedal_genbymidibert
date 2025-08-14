import pandas as pd
import numpy as np


def restore_custom_format(df):
    """
    Restore MIDI DataFrame to a custom flat format where pedal and pitch events
    are separated into explicit rows.

    Steps:
    1. Assign segment IDs based on Bar column changes.
    2. Remove padding rows (Bar=0, Pitch=0, Pedal_1=0).
    3. Keep the first non-zero Bar row at the segment start.
    4. Separate pedal_1 events into individual rows.
    5. Keep pitch rows as separate events.
    6. Merge all events, sort by Position, and reset the index.

    Parameters
    ----------
    df : pd.DataFrame
        Input MIDI DataFrame with multiple columns, including pedal info.

    Returns
    -------
    pd.DataFrame
        Restored DataFrame with columns ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']
    """
    df = df.copy()
    df['seg_id'] = df['Bar'].ne(0).cumsum()

    restored_rows = []

    for seg_id, group in df.groupby('seg_id'):
        # Skip padding rows (Bar=0, Pitch=0, Pedal_1=0)
        is_padding = (
            (group['Bar'] == 0) &
            (group['Pitch'] == 0) &
            (group['Pedal_1'] == 0)
        )
        group = group[~is_padding].copy()

        # Keep the first non-zero Bar row at the start of the segment
        bar_row = group[group['Bar'] != 0].head(1).copy()
        bar_row = bar_row[['Bar', 'Position', 'Pitch', 'Velocity', 'Duration']].copy()
        bar_row['Pedal'] = 0
        restored_rows.append(bar_row)

        # Pedal_1 events
        pedal1_rows = group[group['Pedal_1'] != 0][['Position_1', 'Duration_1', 'Pedal_1']].copy()
        pedal1_rows = pedal1_rows.rename(columns={
            'Position_1': 'Position',
            'Duration_1': 'Duration',
            'Pedal_1': 'Pedal'
        })
        pedal1_rows['Pitch'] = 0
        pedal1_rows['Velocity'] = 0
        pedal1_rows['Bar'] = 0

        # Pitch events
        pitch_rows = group[group['Pitch'] != 0][['Bar', 'Position', 'Pitch', 'Velocity', 'Duration']].copy()
        pitch_rows['Pedal'] = 0

        # Merge pedal and pitch rows, sort by Position
        merged = pd.concat([pedal1_rows, pitch_rows], ignore_index=True)
        merged = merged[['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']]
        merged = merged.sort_values(by='Position').reset_index(drop=True)

        restored_rows.append(merged)

    final_df = pd.concat(restored_rows, ignore_index=True)
    return final_df


def process_bar_events(df):
    """
    Process non-zero Bar events by shifting the Bar=4 to the next row and removing the current row.

    Parameters
    ----------
    df : pd.DataFrame
        MIDI DataFrame after adding pedal events.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated Bar events.
    """
    bar_updates = []  # List of (index, new_value) for Bar updates
    rows_to_remove = []  # Rows to drop after update

    for i in range(len(df)):
        # Skip pedal rows
        if 'Source' in df.columns and df.at[i, 'Source'] == 'pedal':
            continue

        if df.at[i, 'Bar'] != 0:
            next_idx = i + 1
            if next_idx < len(df):
                bar_updates.append((next_idx, 4))
            rows_to_remove.append(i)

    # Apply updates
    for idx, new_value in bar_updates:
        df.at[idx, 'Bar'] = new_value

    # Drop old Bar rows
    if rows_to_remove:
        df = df.drop(index=rows_to_remove)

    return df


def clean_data(df):
    """
    Final data cleaning steps:
    1. If Position value repeats from previous row, replace it with NaN.
    2. Replace all zeros with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after processing Bar events.

    Returns
    -------
    pd.DataFrame
        Cleaned final DataFrame.
    """
    if 'Position' in df.columns:
        for i in range(1, len(df)):
            if pd.notna(df.at[i, 'Position']) and pd.notna(df.at[i - 1, 'Position']):
                if df.at[i, 'Position'] == df.at[i - 1, 'Position']:
                    df.at[i, 'Position'] = np.nan

    columns_to_clean = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']
    for col in columns_to_clean:
        if col in df.columns:
            df[col] = df[col].mask(df[col] == 0, np.nan)

    return df


def process_midi_df(df):
    """
    Complete MIDI DataFrame processing pipeline:
    1. Restore custom flat format (separate pedal/pitch events)
    2. Process Bar events
    3. Clean data and remove temporary columns

    Parameters
    ----------
    df : pd.DataFrame
        Raw MIDI DataFrame.

    Returns
    -------
    pd.DataFrame
        Fully processed DataFrame ready for export.
    """
    df = restore_custom_format(df)
    df = process_bar_events(df)
    df.reset_index(drop=True, inplace=True)
    df = clean_data(df)
    df = df.drop(columns=['Pedal_1', 'Position_1', 'Duration_1', 'Source'], errors='ignore')
    return df


def has_positive_value(row):
    """
    Check if any main feature in the row has a positive value.

    Parameters
    ----------
    row : pd.Series
        DataFrame row

    Returns
    -------
    bool
        True if at least one feature is positive
    """
    for col in ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']:
        try:
            if row[col] > 0:
                return True
        except TypeError:
            continue
    return False


# =============================
# Example usage
# =============================
if __name__ == "__main__":
    input_file = 'output_n.xlsx'
    df = pd.read_excel(input_file)

    processed_df = process_midi_df(df)

    output_file = 'output_nn.xlsx'
    processed_df.to_excel(output_file, index=False)
    print(f"Processing complete! Input: {input_file}, Output: {output_file}")
