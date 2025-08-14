from preprocess import *
from main_full_removal import *

from utils.gen_midi_type_xlsx import process_midi_df, has_positive_value
from utils.gen_predict_midi import apply_business_rules, transform_to_midi

import os


def process_single_file(midi_path):
    """Process a single MIDI file and return structured features for model inference."""
    try:
        # Tokenize the MIDI file
        tokens = tokenizer(midi_path)

        # Extract individual fields from token sequence
        fields = extract_fields_from_tok_sequence(tokens)

        # Map tokens to token IDs
        token_id_dict = build_token_id_dict(tokens, fields)

        # Process key events into structured pairs
        processed_pairs = process_keys(token_id_dict)

        # Convert pairs to initial DataFrame
        df0 = process_pairs(processed_pairs)

        # ===== Additional Pedal processing pipeline =====
        df = process_bar_column(df0)  # Ensure correct Bar column formatting
        df = preprocess_dataframe(df)  # General preprocessing
        df = process_pedal_special_values(df)  # Handle special cases in Pedal data
        # ==============================================

        # Convert DataFrame to 11-dimensional feature vector
        feature_matrix = df_to_feature_vector_11dim(df)
        features = [feature_matrix]

        # Process sequences with sliding window for model input
        processed_data, processed_masks = process_sequences_with_sliding_window(features)

        # Return structured result
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
    Reconstruct the full sequence from overlapping chunks using half-overlap strategy.

    Parameters
    ----------
    processed_data : torch.Tensor
        Model output slices of shape (num_chunks, max_length, channels).
    orig_len : int
        Original sequence length.
    max_length : int
        Length of each chunk.
    stride : int
        Step size for sliding window.
    channels : int
        Number of feature channels.

    Returns
    -------
    torch.Tensor
        Reconstructed sequence of shape (orig_len, channels).
    """
    overlap = max_length - stride
    half = overlap // 2

    idx = 0  # Window index
    device = processed_data.device
    dtype = processed_data.dtype

    # Initialize empty sequence
    seq = torch.zeros((orig_len, channels), dtype=dtype, device=device)

    # Compute all window start positions
    positions = list(range(0, orig_len, stride))
    for win_i, start in enumerate(positions):
        chunk = processed_data[idx]  # shape: (max_length, channels)
        idx += 1

        end = min(start + max_length, orig_len)
        valid_len = end - start

        if win_i == 0:
            # First window, directly assign
            seq[start:end] = chunk[:valid_len]
        else:
            ov = min(overlap, valid_len)
            h = min(half, ov)

            # Assign second half of overlap
            if ov > h:
                seq[start + h: start + ov] = chunk[h: ov]

            # Assign non-overlapping portion
            if valid_len > overlap:
                seq[start + overlap: start + valid_len] = chunk[overlap: valid_len]

    print(seq.shape)
    return seq


def cat(predict_squeezed, input_data):
    """
    Selectively update input features with predicted values and generate a DataFrame for export.

    Parameters
    ----------
    predict_squeezed : torch.Tensor
        Predicted tensor (batch_size, seq_length, 2) representing [Pitch, Duration].
    input_data : torch.Tensor
        Original input tensor (batch_size, seq_length, features).

    Returns
    -------
    updated_input : torch.Tensor
        Input tensor updated with predictions where applicable.
    df1 : pd.DataFrame
        DataFrame containing updated sequences for export.
    exceed_p_count : int
        Number of Pitch values exceeding valid range.
    exceed_d_count : int
        Number of Duration values exceeding valid range.
    """
    idx = [6, 7]  # Channels to update
    pedal_id = [5]  # Pedal channel index

    # Clamp predictions to valid ranges
    p = predict_squeezed[..., 0]
    clipped_p = torch.clamp(p, 2437, 2820)

    d = predict_squeezed[..., 1]
    clipped_d = torch.clamp(d, 157, 2436)

    clipped_pred = torch.stack((clipped_p, clipped_d), dim=1)
    print(clipped_p.shape, clipped_d.shape, clipped_pred.shape)

    # Masks for valid updates
    feature_mask_p = clipped_p != 0
    feature_mask_d = clipped_d != 0
    combined_mask = feature_mask_p & feature_mask_d

    # Update input tensor selectively
    updated_input = input_data.clone()
    updated_input[..., idx] = torch.where(
        combined_mask.unsqueeze(1).expand(-1, 2),
        clipped_pred,
        torch.tensor(0, dtype=updated_input.dtype)
    )

    updated_input[..., pedal_id] = torch.where(
        combined_mask.unsqueeze(-1),
        torch.tensor(2882, dtype=updated_input.dtype),
        torch.tensor(0, dtype=updated_input.dtype)
    )

    # Count values exceeding valid ranges
    exceed_p_count = torch.sum((p < 2437) | (p > 2820) & combined_mask).item()
    exceed_d_count = torch.sum((d < 157) | (d > 2436) & combined_mask).item()

    # Convert updated input to DataFrame
    column_names = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal_1', 'Position_1', 'Duration_1']
    df1 = pd.DataFrame(updated_input.numpy(), columns=column_names)
    print(df1.shape)
    print(f"Number of valid points: {len(df1)}")

    return updated_input, df1, exceed_p_count, exceed_d_count


def inverse_normalize(x):
    """Inverse scale normalized tick IDs back to original range."""
    return torch.round(((x + 1) / 2) * (923 - 157) + 157)


def tick_id_to_duration_tensor(tick_ids):
    """Convert tick IDs to real duration values using piecewise mapping."""
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
    """Assign a condition index (0-4) based on piecewise tick ranges."""
    cond = torch.zeros_like(tick_ids)
    cond += (tick_ids > 539).long()
    cond += (tick_ids > 923).long()
    cond += (tick_ids > 1499).long()
    cond += (tick_ids > 2363).long()
    return cond


def compute_mae_ms_tensor(predict_n, target_n):
    """Compute masked MAE and MSE metrics between predicted and target tick IDs."""
    mask = (target_n != 0).float()
    tick_pred = inverse_normalize(predict_n)
    tick_true = target_n

    tick_pred_real = tick_id_to_duration_tensor(tick_pred)
    tick_true_real = tick_id_to_duration_tensor(tick_true)

    error = (tick_pred_real - tick_true_real).float()
    masked_abs_error = torch.abs(error) * mask
    masked_sqr_error = error ** 2 * mask

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
    # Initialize log records list
    log_records = []

    # Create output directories
    os.makedirs("256mul/output_xlsx_small_deeper", exist_ok=True)
    os.makedirs("256mul/output_midi_small_deeper", exist_ok=True)
    os.makedirs("256mul/original_xlsx_small_deeper", exist_ok=True)

    # Load pretrained model
    best_model_path = "new_model/256mul_7_22_0.5/checkpoints/last.ckpt"
    configuration = BertConfig(
        max_position_embeddings=256,
        position_embedding_type='relative_key_query',
        hidden_size=256,        # Transformer hidden dimension
        num_hidden_layers=6,    # Number of encoder layers
        num_attention_heads=8,  # Attention heads
        intermediate_size=1024, # Feed-forward hidden size
        attn_implementation="eager"
    )
    config_dict = configuration.to_dict()
    best_model = MyLightningModule.load_from_checkpoint(
        best_model_path,
        config=config_dict
    )
    best_model.eval()

    # Process each test MIDI file
    for test_file in data["test"]:
        filename = os.path.splitext(os.path.basename(test_file))[0]

        # Preprocess single file
        result = process_single_file(test_file)
        if result is None:
            continue

        print(f"Original sequence length: {result['original_length']}")

        test_data_n = result['data']

        test_loader = data_loader(
            test_data_n,
            result['mask'],
            shuffle=False
        )

        # Run inference
        all_predictions, all_targets, all_inputs = [], [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                inputs, masks, targets = batch

                # Clamp Pitch values to max range
                targets[..., [2]] = torch.clamp(targets[..., [2]], max=923)

                # Move to model device
                inputs, masks, targets = inputs.to(best_model.device), masks.to(best_model.device), targets.to(best_model.device)

                outputs = best_model(inputs, masks)

                all_predictions.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())
                all_inputs.append(inputs.detach().cpu())

        # Concatenate batch outputs
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)
        inputs = torch.cat(all_inputs)

        # Reconstruct full sequence from sliding windows
        predictions_win = reconstruct_with_half_overlap(predictions, result["original_length"], max_length=256, stride=224, channels=2)
        targets_win = reconstruct_with_half_overlap(targets, result["original_length"], max_length=256, stride=224, channels=3)
        inputs_win = reconstruct_with_half_overlap(inputs, result["original_length"], max_length=256, stride=224, channels=5)

        # Compute loss and accuracy metrics
        test_loss, loss_p, loss_d = best_model.compute_loss(predictions_win, targets_win[..., [1,2]], beta=0.5)
        test_acc, acc_p, acc_d = best_model.compute_accuracy(predictions_win, targets_win[..., [1,2]])
        print(f"Loss: {test_loss}, Pitch Loss: {loss_p}, Duration Loss: {loss_d}")
        print(f"Accuracy: {test_acc}, Pitch Acc: {acc_p}, Duration Acc: {acc_d}")

        # Compute MAE/MSE metrics in milliseconds
        mae_record = compute_mae_ms_tensor(predictions_win[..., [1]], targets_win[..., [2]])

        # Denormalize predictions to tick ranges
        data_p = (2820 - 2437) * (predictions_win[..., [0]] + 1) / 2 + 2437
        data_d = (923 - 157) * (predictions_win[..., [1]] + 1) / 2 + 157
        data_pre = torch.cat((data_p, data_d), dim=1)
        data_tar = targets_win
        data_input = inputs_win

        # Concatenate input and target data for selective updating
        combined_data = torch.cat((data_input, data_tar), dim=1)

        # Update input with predictions and generate DataFrame
        output, df, exceed_p_count, exceed_d_count = cat(data_pre, combined_data)

        # Post-process MIDI DataFrame
        df = process_midi_df(df)

        # Record file-level metrics
        record = {
            "filename": filename,
            "original_shape": str(result['original_shape']),
            "loss": float(test_loss),
            "acc": float(test_acc),
            "exceed_p_count": exceed_p_count,
            "exceed_d_count": exceed_d_count
        }
        record.update(mae_record)
        log_records.append(record)

        # Filter out trailing rows without positive values
        valid_rows = pd.Series([False] * len(df))
        for j in range(len(df) - 1, -1, -1):
            if has_positive_value(df.iloc[j]):
                valid_rows[:j + 1] = True
                break
        df = df[valid_rows]

        # Save output Excel
        xlsx_path = os.path.join("256mul/output_xlsx_small_deeper", f"{filename}_output.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"Output Excel saved: {xlsx_path}")

        # Save original Excel
        xlsx_path_ori = os.path.join("256mul/original_xlsx_small_deeper", f"{filename}_original.xlsx")
        df_ori = pd.DataFrame(result["ori"], columns=['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal'])
        df_ori.to_excel(xlsx_path_ori, index=False)
        print(f"Original Excel saved: {xlsx_path_ori}")

        # Generate MIDI file
        processed_df = apply_business_rules(df)
        midi_path = os.path.join("256mul/output_midi_small_deeper", f"{filename}_output.mid")
        transform_to_midi(processed_df, midi_path=midi_path)
        print(f"MIDI file generated: {midi_path}")

    # Save full log
    log_df = pd.DataFrame(log_records)
    log_df.to_excel("256mul/log_small_deeper.xlsx", index=False)
    print("All logs saved to '256mul/log_small_deeper.xlsx'")
