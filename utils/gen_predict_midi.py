import pandas as pd
from miditok import REMI, TokenizerConfig


def process_music_df(raw_df):
    """
    Complete music data processing: cleaning and converting to standard format.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Input raw music data, usually read from Excel.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with numeric columns and standardized values.
    """
    required_columns = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']

    # Verify required columns exist
    missing_cols = [col for col in required_columns if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # Keep only required columns
    df = raw_df[required_columns].copy()

    # Data cleaning: replace empty or invalid strings with NA
    df = df.replace(['', 'NULL', 'NA'], pd.NA)
    numeric_cols = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Data conversion and standardization
    df['Bar'] = df['Bar'].fillna(0).astype(int)
    df['Position'] = df['Position'].ffill().astype('int32')
    # Map special pedal values to pitches
    df.loc[df['Pedal'] == 2882, 'Pitch'] = 93
    df.loc[df['Pedal'] == 1019, 'Pitch'] = 94
    df['Pitch'] = df['Pitch'].fillna(pd.NA)
    df['Velocity'] = df['Velocity'].fillna(0).astype(int)

    return df


def apply_business_rules(df):
    """
    Apply additional business rules to the processed DataFrame:
    - Map certain pitches to pedal codes
    - Remove zero values
    - Prevent repeated positions

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame after initial cleaning.

    Returns
    -------
    pd.DataFrame
        DataFrame with business rules applied and integer type columns.
    """
    df = df.replace(['', 'NULL', 'NA'], pd.NA).copy()
    numeric_cols = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    df['Bar'] = df['Bar'].replace(0, pd.NA)
    mask_314 = df['Pitch'] == 93
    mask_315 = df['Pitch'] == 94
    df.loc[mask_314, 'Pedal'] = 2882
    df.loc[mask_315, 'Pedal'] = 2883
    df['Pitch'] = df['Pitch'].where(~mask_314 & ~mask_315, pd.NA)
    df['Velocity'] = df['Velocity'].replace(0, pd.NA)

    # Remove duplicate Position values
    position_dupe_mask = df['Position'] == df['Position'].shift(1)
    df.loc[position_dupe_mask, 'Position'] = pd.NA
    df['Position'] = df['Position'].astype('Int32')

    # Ensure integer type for all numeric columns
    int_cols = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']
    df[int_cols] = df[int_cols].astype('Int32')

    return df


def transform_to_midi(df, midi_path):
    """
    Convert processed DataFrame to a MIDI file using REMI tokenizer.

    Special handling:
    - Certain pedal codes (2882, 2883) are swapped with the previous value in sequence.
    - Decoded sequence is converted and exported as a MIDI file.

    Parameters
    ----------
    df : pd.DataFrame
        Processed music DataFrame.
    midi_path : str
        Output path for the generated MIDI file.
    """
    ids = []
    for _, row in df.iterrows():
        for item in row:
            if pd.notna(item) and isinstance(item, (int, float)):
                item_int = int(item)
                # Swap position if item is a special pedal code
                if (item_int == 2882 or item_int == 2883) and ids:
                    prev = ids[-1]
                    ids[-1] = item_int
                    ids.append(prev)
                else:
                    ids.append(item_int)

    # Tokenizer configuration
    config1 = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={
            (0, 4): 96,
            (4, 12): 48,
            (12, 36): 24,
            (36, 108): 12,
            (108, 120): 6,
        },
        num_velocities=64,
        use_sustain_pedals=True,
        sustain_pedal_duration=True,
        num_tempos=128,
        tempo_range=(40, 250),
    )
    tokenizer = REMI(config1)

    # Decode token IDs and export MIDI
    decoded_music = tokenizer.decode([ids])
    decoded_music.dump_midi(midi_path)


def main(file_path):
    """
    Main pipeline:
    1. Read Excel file into DataFrame
    2. Process and clean music data
    3. Apply business rules
    4. Convert and save as MIDI file
    """
    raw_df = pd.read_excel(file_path, engine='openpyxl')
    print("Step 1: Excel loaded.")

    processed_df = process_music_df(raw_df)
    processed_df = apply_business_rules(processed_df)

    transform_to_midi(processed_df, "./output_cat_4.midi")
    print("Processing complete. MIDI file generated.")


if __name__ == "__main__":
    main("../output/output_n1.xlsx")
