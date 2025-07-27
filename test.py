from preprocess import *
from main_0619 import *
from utils.b_reverse_fill import has_positive_value
from utils.gen_midi_type_xlsx import process_midi_df
from utils.a_midi_gen_new import apply_business_rules, transform_to_midi

import numpy as np

import os

def process_single_file(midi_path):
    """处理单个MIDI文件并返回结果"""
    try:

        tokens = tokenizer(midi_path)
        fields = extract_fields_from_tok_sequence(tokens)
        token_id_dict = build_token_id_dict(tokens, fields)
        processed_pairs = process_keys(token_id_dict)

        df0 = process_pairs(processed_pairs)

        # ===== 新增的 Pedal 处理流程 =====
        df = process_bar_column(df0)
        df = preprocess_dataframe(df)
        df = process_pedal_special_values(df)
        # ==============================

        feature_matrix = df_to_feature_vector_11dim(df)
        features=[]
        features.append(feature_matrix)
        processed_data, processed_masks = process_sequences_with_sliding_window(features)


        # 6. 返回结果
        return {
            "ori":df0,
            "data": processed_data,
            "mask": processed_masks,
            "filename": os.path.basename(midi_path),
            "original_shape": feature_matrix.shape,
            "original_length": feature_matrix.shape[1]
        }

    except Exception as e:
        print(f"处理文件 {midi_path} 时出错: {str(e)}")
        return None


def reconstruct_with_half_overlap(processed_data: torch.Tensor, orig_len: int,
                                  max_length=256, stride=224, channels=8):
    """
    使用半重叠的方式重构序列，适用于 PyTorch 张量类型（float32）。

    参数
    ----
    processed_data : torch.Tensor
        模型输出的切片，shape = (num_chunks, max_length, channels)，dtype=torch.float32。
    orig_len : int
        原始序列长度。
    max_length : int
        每个切片的长度。
    stride : int
        每次滑动的步长。
    channels : int
        特征通道数。

    返回
    ----
    reconstructed : list[torch.Tensor]
        一个长度为1的列表，包含重构后的序列，shape = (orig_len, channels)
    """
    overlap = max_length - stride
    half = overlap // 2


    idx = 0  # 切片游标
    device = processed_data.device
    dtype = processed_data.dtype

    # 初始化空序列
    seq = torch.zeros((orig_len, channels), dtype=dtype, device=device)

    # 计算所有窗口起始点
    positions = list(range(0, orig_len, stride))
    for win_i, start in enumerate(positions):
        chunk = processed_data[idx]  # shape: (max_length, channels)
        idx += 1

        end = min(start + max_length, orig_len)
        valid_len = end - start

        if win_i == 0:
            # 第一段直接赋值（可能被截断）
            seq[start:end] = chunk[:valid_len]
        else:
            ov = min(overlap, valid_len)
            h = min(half, ov)

            # 后半重叠
            if ov > h:
                seq[start + h : start + ov] = chunk[h : ov]

            # 非重叠区域
            if valid_len > overlap:
                seq[start + overlap : start + valid_len] = chunk[overlap : valid_len]


    print(seq.shape)
    return seq




def cat(predict_squeezed, loss_mask1, input_data):
    """
    选择性地将predict中非零数据更新到input中，然后导出筛选数据到Excel

    参数:
    - predict: 预测值张量 (batch_size, seq_length, 1)
    - loss_mask: 损失掩码 (batch_size, seq_length, features)
    - input_data: 输入数据 (batch_size, seq_length, features)
    - excel_path: Excel保存路径

    返回:
    - updated_input: 更新后的输入数据
    - concatenated_data: 拼接的有效数据
    """
    idx = 7  # 指定要更新的通道索引

    # 数据预处理
    #predict_squeezed = predict.squeeze(-1)  # 去除单维度通道 (batch, seq)(n, 256, 1)->(n, 256)

    x = input_data[..., 5].float()  # 特征通道 (batch, seq)
    base_mask = loss_mask1

    # 生成动态掩码：确定predict中需要保留的非零位置
    # 仅在特征有意义(1-x>0)且预测非零的位置更新
    clipped_pred = torch.clamp(predict_squeezed, min=157, max=2436)# 应急修改将输出范围限定

    feature_significant_mask = x != 0

    # 组合掩码：基础掩码和动态掩码的交集
    combined_mask = base_mask.bool() & feature_significant_mask

    # 选择性更新：只更新非零数据的位置
    updated_input = input_data.clone()
    # 只更新掩码为True的位置
    updated_input[..., idx] = torch.where(
        combined_mask,
        clipped_pred,
        input_data[..., idx].float()  # 在其他位置保持原始值
    )

    #检测超出范围的次数
    exceed_mask = combined_mask & ((predict_squeezed < 157) | (predict_squeezed > 2436))
    # print(predict_squeezed.shape)
    exceed_count = torch.sum(exceed_mask).item()

    # 应用基础掩码筛选有效数据
    #batch_size = updated_input.size(0)
    #valid_data_list = []

    #for mask_i in range(batch_size):
        # 获取当前批次的基础掩码
    #    batch_mask = base_mask[mask_i].bool()
    #    if batch_mask.any():
            # 获取基础掩码为True的有效数据
    #        valid_batch_data = updated_input[mask_i, batch_mask]
    #        valid_data_list.append(valid_batch_data)

    #if not valid_data_list:
    #    print("警告: 没有有效数据点!")
    #    return None

    # 拼接所有有效数据 (n, 6)
    #concatenated_data = torch.cat(valid_data_list, dim=0)
    #concatenated_data = concatenated_data.reshape(-1, 8)
    # 准备导出到Excel
    column_names = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal_1', 'Position_1', 'Duration_1']

    # 创建DataFrame
    df1 = pd.DataFrame(updated_input.numpy(), columns=column_names)
    print(df1.shape)

    print(f"有效数据点数量: {len(df1)}")

    return updated_input, df1, exceed_count

def inverse_normalize(x):
    return torch.round(((x + 1) / 2) * (923 - 157) + 157)

def tick_id_to_duration_tensor(tick_ids):
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
    cond = torch.zeros_like(tick_ids)
    cond += (tick_ids > 539).long()
    cond += (tick_ids > 923).long()
    cond += (tick_ids > 1499).long()
    cond += (tick_ids > 2363).long()
    return cond  # 返回0~4之间的整数，分别代表cond1~cond5


def compute_mae_ms_tensor(predict_n, target_n):
    # 反归一化为 tick ID
    mask = (target_n != 0).float()
    tick_pred = inverse_normalize(predict_n)
    tick_true = target_n

    # 映射为真实时长（tick-duration）
    tick_pred_real = tick_id_to_duration_tensor(tick_pred)
    tick_true_real = tick_id_to_duration_tensor(tick_true)

    # 计算 MAE 和 MSE，仅保留 mask>0 的位置
    error = (tick_pred_real - tick_true_real).float()
    abs_error = torch.abs(error)
    sqr_error = error ** 2

    masked_abs_error = abs_error * mask
    masked_sqr_error = sqr_error * mask

    total_valid = mask.sum()
    mae_tick = masked_abs_error.sum() / (total_valid + 1e-8)
    mse_tick = masked_sqr_error.sum() / (total_valid + 1e-8)

    # 统计转移情况
    cond_pred = get_condition_index(tick_pred)
    cond_true = get_condition_index(tick_true)

    record = {
        "mae_ms": mae_tick.item() * 5.21,
        "mse_ms": mse_tick.item() * (5.21 ** 2),
        "total_valid": total_valid.item()
    }

    for i in range(5):
        for j in range(5):
            key = f"{i+1}_{j+1}"
            count = ((cond_pred == i) & (cond_true == j) & (mask > 0)).sum().item()
            record[key] = count

    return record




if __name__ == "__main__":
    # 加载预处理的测试数据
    log_records = []

    os.makedirs("256/output_xlsx_small_deeper", exist_ok=True)
    os.makedirs("256/output_midi_small_deeper", exist_ok=True)
    os.makedirs("256/original_xlsx_small_deeper", exist_ok=True)

    # 加载模型
    best_model_path = "new_model/256_7_22/checkpoints/epoch_24-val_loss_0.04.ckpt"
    configuration = BertConfig(
        max_position_embeddings=256,
        position_embedding_type='relative_key_query',
        hidden_size=256,  # 128
        num_hidden_layers=6,  # 4
        num_attention_heads=8,  # 4
        intermediate_size=1024,  # 128
        attn_implementation="eager"
    )
    config_dict = configuration.to_dict()
    best_model = MyLightningModule.load_from_checkpoint(
        best_model_path,
        config=config_dict
    )
    best_model.eval()



    # 在文件处理循环中
    for test_file in data["test"]:
        filename = os.path.splitext(os.path.basename(test_file))[0]
        result = process_single_file(test_file)
        print(result['original_length'])

        if result is None:
            continue

        #test_data_n = normalize(result['data'])
        test_data_n = result['data']

        test_loader = data_loader(
            test_data_n,
            result['mask'],
            shuffle=False
        )


        # 进行推理
        all_predictions = []
        all_targets = []
        all_inputs = []
        all_masks = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                inputs, masks, targets = batch

                targets = torch.clamp(targets, max=923)

                inputs = inputs.to(best_model.device)
                masks = masks.to(best_model.device)
                targets = targets.to(best_model.device)

                outputs = best_model(inputs, masks)
                # print(outputs.shape)
                # print(inputs.shape)
                # print(masks.shape)

                all_predictions.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())
                all_masks.append(masks.detach().cpu())
                all_inputs.append(inputs.detach().cpu())

        predictions = torch.cat(all_predictions)
        print(predictions.shape)
        targets = torch.cat(all_targets).unsqueeze(2)
        print(targets.shape)
        inputs = torch.cat(all_inputs)
        print(inputs.shape)
        masks = torch.cat(all_masks).unsqueeze(2)
        print(masks.shape)

        predictions_win = reconstruct_with_half_overlap(predictions, result["original_length"],
                                                   max_length=256, stride=224, channels=1)
        targets_win = reconstruct_with_half_overlap(targets, result["original_length"],
                                                   max_length=256, stride=224, channels=1)
        inputs_win = reconstruct_with_half_overlap(inputs, result["original_length"],
                                                   max_length=256, stride=224, channels=7)
        masks_win = reconstruct_with_half_overlap(masks, result["original_length"],
                                                   max_length=256, stride=224, channels=1)

        # 计算测试指标
        test_loss = best_model.compute_loss(predictions_win, targets_win.squeeze(), beta=0.9)

        test_acc = best_model.compute_accuracy(predictions_win, targets_win.squeeze())
        print(test_loss, test_acc)

        mae_record = compute_mae_ms_tensor(predictions_win, targets_win)
        data_pre = (923 - 157) * (predictions_win + 1) / 2 + 157
        data_tar = targets_win


        data_input = inputs_win
        data_masks = masks_win

        #print(data_pre.numpy())

        #data_tar_n = data_tar.unsqueeze(2)  # 形状变为 (n, 256, 1)
        c_data = torch.cat((data_input, data_tar), dim=1)  # 形状 (n, 8)


        output, df, exceed_count = cat(data_pre.squeeze(), data_masks.squeeze(), c_data)


        df = process_midi_df(df)

        record = {
            "filename": filename,
            "original_shape": str(result['original_shape']),
            "loss": test_loss.float(),
            "acc": test_acc.float(),
            "exceed_count": exceed_count
        }
        record.update(mae_record)
        log_records.append(record)

        valid_rows = pd.Series([False] * len(df))
        for j in range(len(df) - 1, -1, -1):
            row = df.iloc[j]
            if has_positive_value(row):
                valid_rows[:j + 1] = True
                break

        df = df[valid_rows]

        # 保存Excel
        xlsx_path = os.path.join("256/output_xlsx_small_deeper", f"{filename}_output.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"Excel文件已保存: {xlsx_path}")

        # 保存Excel
        xlsx_path_ori = os.path.join("256/original_xlsx_small_deeper", f"{filename}_original.xlsx")
        column_names = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']

        # 创建DataFrame
        df1 = pd.DataFrame(result["ori"], columns=column_names)
        df1.to_excel(xlsx_path_ori, index=False)
        print(f"Excel文件已保存: {xlsx_path_ori}")

        # 生成MIDI
        processed_df = apply_business_rules(df)
        midi_path = os.path.join("256/output_midi_small_deeper", f"{filename}_output.mid")
        transform_to_midi(processed_df, midi_path=midi_path)
        print(f"MIDI文件已生成: {midi_path}")


    # 保存日志
    log_df = pd.DataFrame(log_records)
    log_df.to_excel("256/log_small_deeper.xlsx", index=False)






