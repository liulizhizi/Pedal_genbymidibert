from preprocess import *
from main_partial_removal import *

from utils.gen_midi_type_xlsx import process_midi_df, has_positive_value
from utils.gen_predict_midi import apply_business_rules, transform_to_midi

import os


def process_single_file(midi_path):
    """
    Process a single MIDI file and return structured results.

    Workflow:
    1. Tokenize the MIDI file using the REMI tokenizer.
    2. Convert tokens into a DataFrame with fields like Bar, Position, Pitch, etc.
    3. Handle special Pedal events (2882 represents sustain pedal events).
    4. Convert the DataFrame into an 11-dimensional feature matrix.
    5. Slice the feature matrix using a sliding window and generate corresponding masks.

    Parameters:
    - midi_path: str, path to the MIDI file

    Returns:
    - dict containing:
        - "ori": original DataFrame
        - "data": sliced feature matrix (sliding window)
        - "mask": corresponding mask
        - "filename": file name
        - "original_shape": shape of the original feature matrix
        - "original_length": length of the original feature matrix
    """
    try:
        tokens = tokenizer(midi_path)
        fields = extract_fields_from_tok_sequence(tokens)
        token_id_dict = build_token_id_dict(tokens, fields)
        processed_pairs = process_keys(token_id_dict)

        # Convert token pairs to DataFrame
        df0 = process_pairs(processed_pairs)

        # ===== Pedal event processing =====
        df = process_bar_column(df0)
        df = preprocess_dataframe(df)
        df = process_pedal_special_values(df)
        # ================================

        # Convert to 11-dimensional feature matrix
        feature_matrix = df_to_feature_vector_11dim(df)
        features = [feature_matrix]

        # Apply sliding window slicing with mask generation
        processed_data, processed_masks = process_sequences_with_sliding_window(features)

        # Return results
        return {
            "ori": df0,
            "data": processed_data,
            "mask": processed_masks,
            "filename": os.path.basename(midi_path),
            "original_shape": feature_matrix.shape,
            "original_length": feature_matrix.shape[1]
        }

    except Exception as e:
        print(f"Error processing file {midi_path}: {str(e)}")
        return None


def reconstruct_with_half_overlap(processed_data: torch.Tensor, orig_len: int,
                                  max_length=256, stride=224, channels=8):
    """
    Reconstruct the full sequence from sliced windows using half-overlap.

    Suitable for PyTorch tensors output by the model. Handles overlapping regions correctly.

    Parameters:
    - processed_data: torch.Tensor, shape=(num_chunks, max_length, channels)
        Sliced windows of the sequence.
    - orig_len: int, original sequence length
    - max_length: int, length of each window
    - stride: int, sliding step size
    - channels: int, number of feature channels

    Returns:
    - seq: torch.Tensor, reconstructed full sequence, shape=(orig_len, channels)
    """
    overlap = max_length - stride
    half = overlap // 2
    idx = 0
    device = processed_data.device
    dtype = processed_data.dtype

    seq = torch.zeros((orig_len, channels), dtype=dtype, device=device)
    positions = list(range(0, orig_len, stride))

    for win_i, start in enumerate(positions):
        chunk = processed_data[idx]
        idx += 1
        end = min(start + max_length, orig_len)
        valid_len = end - start

        if win_i == 0:
            seq[start:end] = chunk[:valid_len]
        else:
            ov = min(overlap, valid_len)
            h = min(half, ov)
            if ov > h:
                seq[start + h: start + ov] = chunk[h: ov]
            if valid_len > overlap:
                seq[start + overlap: start + valid_len] = chunk[overlap: valid_len]

    print(seq.shape)
    return seq


def cat(predict_squeezed, loss_mask1, input_data):
    """
    Selectively update a specific channel of input data using predictions
    and export filtered data as a DataFrame.

    Logic:
    - Update channel `idx=7` based on mask and predicted values.
    - Clamp predictions to a valid range (157-2436).
    - Generate the final DataFrame including original features and Pedal info.

    Parameters:
    - predict_squeezed: torch.Tensor, predicted values, shape=(batch, seq)
    - loss_mask1: torch.Tensor, base mask, shape=(batch, seq)
    - input_data: torch.Tensor, input features, shape=(batch, seq, features)

    Returns:
    - updated_input: updated input tensor
    - df1: corresponding DataFrame, with Pedal information
    - exceed_count: number of predictions outside the valid range
    """
    idx = 7  # feature channel to update
    x = input_data[..., 5].float()  # Pedal channel
    base_mask = loss_mask1

    clipped_pred = torch.clamp(predict_squeezed, min=157, max=2436)
    feature_significant_mask = x != 0
    combined_mask = base_mask.bool() & feature_significant_mask

    updated_input = input_data.clone()
    updated_input[..., idx] = torch.where(
        combined_mask,
        clipped_pred,
        input_data[..., idx].float()
    )

    exceed_mask = combined_mask & ((predict_squeezed < 157) | (predict_squeezed > 2436))
    exceed_count = torch.sum(exceed_mask).item()

    column_names = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration',
                    'Pedal_1', 'Position_1', 'Duration_1']
    df1 = pd.DataFrame(updated_input.numpy(), columns=column_names)
    print(df1.shape)
    print(f"Number of valid data points: {len(df1)}")

    return updated_input, df1, exceed_count


def inverse_normalize(x):
    """Convert values in [-1, 1] back to the original tick ID range."""
    return torch.round(((x + 1) / 2) * (923 - 157) + 157)


def tick_id_to_duration_tensor(tick_ids):
    """
    Map tick IDs to actual duration in ticks.

    Different intervals are handled according to the empirical formula.
    """
    tick_ids = tick_ids.round().long()
    result = torch.zeros_like(tick_ids)

    cond1 = tick_ids <= 539
    cond2 = (tick_ids > 539) & (tick_ids <= 923)
    cond3 = (tick_ids > 923) & (tick_ids <= 1499)
    cond4 = (tick_ids > 1499) & (tick_ids <= 2363)
    cond5 = (tick_ids > 2363) & (tick_ids <= 2436)

    result[cond1] = tick_ids[cond1]
    result[cond2] = tick_ids[cond2] * 2 - 539
    result[cond3] = tick_ids[cond3] * 4 - 923 * 2 + 539
    result[cond4] = tick_ids[cond4] * 8 - 1499 * 4 + 923 * 2 - 539
    result[cond5] = tick_ids[cond5] * 16 - 2363 * 8 + 1499 * 4 - 923 * 2 + 539

    return result


def get_condition_index(tick_ids):
    """
    Return the interval index of tick IDs.
    0-4 correspond to five different tick intervals.
    """
    cond = torch.zeros_like(tick_ids)
    cond += (tick_ids > 539).long()
    cond += (tick_ids > 923).long()
    cond += (tick_ids > 1499).long()
    cond += (tick_ids > 2363).long()
    return cond


def compute_mae_ms_tensor(predict_n, target_n):
    """
    Compute MAE and MSE between predictions and ground truth,
    and count the number of transitions for each interval condition.

    Returns a dictionary containing:
    - mae_ms: Mean Absolute Error in milliseconds
    - mse_ms: Mean Squared Error in milliseconds
    - total_valid: number of valid samples
    - 1_1, 1_2, ... 5_5: counts of predicted vs true interval transitions
    """
    mask = (target_n != 0).float()
    tick_pred = inverse_normalize(predict_n)
    tick_true = target_n

    tick_pred_real = tick_id_to_duration_tensor(tick_pred)
    tick_true_real = tick_id_to_duration_tensor(tick_true)

    error = (tick_pred_real - tick_true_real).float()
    abs_error = torch.abs(error)
    sqr_error = error ** 2

    masked_abs_error = abs_error * mask
    masked_sqr_error = sqr_error * mask

    total_valid = mask.sum()
    mae_tick = masked_abs_error.sum() / (total_valid + 1e-8)
    mse_tick = masked_sqr_error.sum() / (total_valid + 1e-8)

    cond_pred = get_condition_index(tick_pred)
    cond_true = get_condition_index(tick_true)

    record = {
        "mae_ms": mae_tick.item() * 5.21,
        "mse_ms": mse_tick.item() * (5.21 ** 2),
        "total_valid": total_valid.item()
    }

    for i in range(5):
        for j in range(5):
            key = f"{i + 1}_{j + 1}"
            count = ((cond_pred == i) & (cond_true == j) & (mask > 0)).sum().item()
            record[key] = count

    return record


if __name__ == "__main__":
    # Initialize a log to store metrics for each file
    log_records = []

    # Create output directories if they do not exist
    os.makedirs("256_Partial/output_xlsx", exist_ok=True)
    os.makedirs("256_Partial/output_midi", exist_ok=True)
    os.makedirs("256_Partial/original_xlsx", exist_ok=True)

    # Load the trained model checkpoint
    best_model_path = "new_model/256_7_22/checkpoints/epoch_24-val_loss_0.04.ckpt"
    configuration = BertConfig(
        max_position_embeddings=256,
        position_embedding_type='relative_key_query',
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        attn_implementation="eager"
    )
    config_dict = configuration.to_dict()
    best_model = MyLightningModule.load_from_checkpoint(
        best_model_path,
        config=config_dict
    )
    best_model.eval()  # set to evaluation mode

    # Iterate over all test files
    for test_file in data["test"]:
        filename = os.path.splitext(os.path.basename(test_file))[0]

        # Process the MIDI file to get features, masks, and original DataFrame
        result = process_single_file(test_file)
        if result is None:
            continue

        print(f"Original sequence length: {result['original_length']}")

        test_data_n = result['data']

        # Create a DataLoader for the processed sequences
        test_loader = data_loader(
            test_data_n,
            result['mask'],
            shuffle=False
        )

        # Run inference and collect predictions
        all_predictions = []
        all_targets = []
        all_inputs = []
        all_masks = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                inputs, masks, targets = batch

                # Clamp target values to valid range
                targets = torch.clamp(targets, max=923)

                # Move tensors to model device
                inputs = inputs.to(best_model.device)
                masks = masks.to(best_model.device)
                targets = targets.to(best_model.device)

                # Forward pass
                outputs = best_model(inputs, masks)

                all_predictions.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())
                all_masks.append(masks.detach().cpu())
                all_inputs.append(inputs.detach().cpu())

        # Concatenate batch outputs
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets).unsqueeze(2)
        inputs = torch.cat(all_inputs)
        masks = torch.cat(all_masks).unsqueeze(2)

        # Reconstruct full sequences from overlapping windows
        predictions_win = reconstruct_with_half_overlap(
            predictions, result["original_length"], max_length=256, stride=224, channels=1
        )
        targets_win = reconstruct_with_half_overlap(
            targets, result["original_length"], max_length=256, stride=224, channels=1
        )
        inputs_win = reconstruct_with_half_overlap(
            inputs, result["original_length"], max_length=256, stride=224, channels=7
        )
        masks_win = reconstruct_with_half_overlap(
            masks, result["original_length"], max_length=256, stride=224, channels=1
        )

        # Compute loss and accuracy
        test_loss = best_model.compute_loss(predictions_win, targets_win.squeeze(), beta=0.9)
        test_acc = best_model.compute_accuracy(predictions_win, targets_win.squeeze())
        print(f"Loss: {test_loss}, Accuracy: {test_acc}")

        # Compute detailed MAE/MSE metrics for Pedal durations
        mae_record = compute_mae_ms_tensor(predictions_win, targets_win)

        # Prepare data for selective channel update
        data_pre = (923 - 157) * (predictions_win + 1) / 2 + 157
        data_tar = targets_win
        data_input = inputs_win
        data_masks = masks_win

        # Concatenate inputs and targets along features
        c_data = torch.cat((data_input, data_tar), dim=1)

        # Selectively update Pedal channel using predictions
        output, df, exceed_count = cat(data_pre.squeeze(), data_masks.squeeze(), c_data)

        # Post-process DataFrame for MIDI export
        df = process_midi_df(df)

        # Log metrics for this file
        record = {
            "filename": filename,
            "original_shape": str(result['original_shape']),
            "loss": test_loss.float(),
            "acc": test_acc.float(),
            "exceed_count": exceed_count
        }
        record.update(mae_record)
        log_records.append(record)

        # Keep only rows with positive values
        valid_rows = pd.Series([False] * len(df))
        for j in range(len(df) - 1, -1, -1):
            row = df.iloc[j]
            if has_positive_value(row):
                valid_rows[:j + 1] = True
                break
        df = df[valid_rows]

        # Save the processed DataFrame as Excel
        xlsx_path = os.path.join("256_Partial/output_xlsx", f"{filename}_output.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"Excel file saved: {xlsx_path}")

        # Save original MIDI DataFrame as Excel
        xlsx_path_ori = os.path.join("256_Partial/original_xlsx", f"{filename}_original.xlsx")
        column_names = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']
        df1 = pd.DataFrame(result["ori"], columns=column_names)
        df1.to_excel(xlsx_path_ori, index=False)
        print(f"Original Excel file saved: {xlsx_path_ori}")

        # Generate MIDI file from processed DataFrame
        processed_df = apply_business_rules(df)
        midi_path = os.path.join("256_Partial/output_midi", f"{filename}_output.mid")
        transform_to_midi(processed_df, midi_path=midi_path)
        print(f"MIDI file generated: {midi_path}")

        break

    # Save overall log to Excel
    log_df = pd.DataFrame(log_records)
    log_df.to_excel("256_Partial/log.xlsx", index=False)
    print("Processing completed. Log saved.")
