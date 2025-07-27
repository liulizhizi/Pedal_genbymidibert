from miditok import REMI
import json
import pandas as pd
from collections import defaultdict
import numpy as np
import os

from sympy.codegen.ast import continue_

# 加载分词器配置和初始化
tokenizer_path = "./tokenizer_hi2.json"
tokenizer = REMI(params=tokenizer_path)

with open('maestro_splits.json', 'r') as f:
    data = json.load(f)

train_files = data["train"]
val_files = data["validation"]
test_files = data["test"]


# 提取字段函数
def extract_fields_from_tok_sequence(tok_sequence):
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
    return list(zip(tok_sequence[0].tokens, fields["ids"]))


def process_keys(pairs):
    return [(key.split('_')[0], value) for key, value in pairs]


def process_pairs(pairs):
    fixed_columns = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']
    rows = []
    current_row = {}

    for key, value in pairs:
        current_row[key] = value
        if key == 'Duration':
            for col in fixed_columns:
                if col not in current_row:
                    current_row[col] = None
            filtered_row = {k: v for k, v in current_row.items() if k in fixed_columns}
            rows.append(filtered_row)
            current_row = {}

    return pd.DataFrame(rows, columns=fixed_columns)


def preprocess_dataframe(df):
    df = df.reset_index(drop=True)
    df['Bar'] = df['Bar'].fillna(0).astype(int)
    # 将 Bar 为 4 的行依次编号为 1, 2, ...
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

# 处理 Bar 列：插入数据行和空行
def process_bar_column(df):
    """处理Bar列：插入新行并修改原行（空白行改为补0）"""
    new_df = pd.DataFrame(columns=df.columns)

    for _, row in df.iterrows():
        current_bar = row['Bar']

        if pd.notna(current_bar) and current_bar != 0:
            # 创建数据行（当前段标记）
            data_row = pd.Series(0, index=df.columns)
            data_row['Bar'] = current_bar

            # 创建补0行（原来是NaN）
            zero_row = pd.Series(0, index=df.columns)

            # 添加段头和补0行
            temp_df = pd.DataFrame([data_row, zero_row])
            new_df = pd.concat([new_df, temp_df], ignore_index=True)

            # 修改当前行的Bar为0（表示段内事件）
            row = row.copy()
            row['Bar'] = 0

        # 添加当前数据行
        new_df = pd.concat([new_df, row.to_frame().T], ignore_index=True)

    return new_df


def process_pedal_special_values(df, max_len=10):
    """
    处理 Pedal 特殊值（2882 和 1563），并对每段长度超过 max_len 的段拆分子段。
    每个子段（从第二个开始）在开头插入两行空行用于存储 Pedal 信息。

    参数:
    - df: 原始 DataFrame，需包含 ['Bar','Position','Pitch','Velocity','Duration','Pedal']。
    - max_len: 每个子段的最大“非事件”行数，默认 10。

    返回:
    - result: 处理后的 DataFrame，包含 ['Pedal_1','Position_1','Duration_1','Pedal_2','Position_2','Duration_2']。
    """
    df = df.copy()
    df.fillna(0, inplace=True)

    # 初始化新的列
    df['Pedal_1'] = df['Pedal'].where(df['Pedal'] == 2882, 0)
    df['Position_1'] = df['Position'].where(df['Pedal'] == 2882, 0)
    df['Duration_1'] = df['Duration'].where(df['Pedal'] == 2882, 0)
    #df['Pedal_2'] = df['Pedal'].where(df['Pedal'] == 1563, 0)
    #df['Position_2'] = df['Position'].where(df['Pedal'] == 1563, 0)
    #df['Duration_2'] = df['Duration'].where(df['Pedal'] == 1563, 0)

    # 标记大段：Bar != 0 为新段
    df['seg_id'] = df['Bar'].ne(0).cumsum()
    df = df.reset_index(drop=True)
    df['row_idx'] = df.index

    all_result = []

    for seg_id, seg_df in df.groupby('seg_id'):
        seg_df = seg_df.reset_index(drop=True)
        seg_df['local_idx'] = seg_df.index

        # 提取事件和非事件
        #events = seg_df[seg_df['Pedal'].isin([2882, 1563])].copy()
        events = seg_df[seg_df['Pedal'].isin([2882])].copy()
        contents = seg_df[seg_df['Pedal'] == 0].copy()
        contents = contents.reset_index(drop=True)
        contents['content_idx'] = contents.index  # 关键：非事件行在组内的索引

        # 创建 local_idx 到 content_idx 的映射
        local_to_content = contents.set_index('local_idx')['content_idx'].to_dict()

        # 拆分 contents 为子段（确保末尾有 buffer 子段）
        num_chunks = (len(contents) // max_len) + 1

        subseg_list = []

        for i in range(num_chunks):
            start = i * max_len
            end = start + max_len
            chunk = contents.iloc[start:end].copy()

            # 插入两行空白用于事件填充（从第二段起）
            if i > 0:
                blank = pd.DataFrame(0, index=range(2), columns=chunk.columns)
                blank['Bar'] = 0
                chunk = pd.concat([blank, chunk], ignore_index=True)

            chunk['subseg_id'] = f"{seg_id}_{i}"
            subseg_list.append(chunk)

        # 修正事件分配逻辑：使用 content_idx 计算块ID
        def assign_subseg_id(row):
            # 找到该事件之前最近的 content row（按 local_idx 对应）
            cond = (contents['local_idx'] <= row['local_idx'])

            if not cond.any():
                return f"{seg_id}_0"  # 没有前驱非事件行时分配到第一个块

            nearest_local = contents.loc[cond, 'local_idx'].max()
            nearest_content_idx = local_to_content.get(nearest_local, 0)
            chunk_id = nearest_content_idx // max_len
            return f"{seg_id}_{int(chunk_id)}"

        events['subseg_id'] = events.apply(assign_subseg_id, axis=1)
        subseg_list.append(events)

        merged = pd.concat(subseg_list, ignore_index=True)
        all_result.append(merged)

    merged_all = pd.concat(all_result, ignore_index=True)

    # 仅保留非事件行用于填充
    result = merged_all[merged_all['Pedal'] == 0].copy()

    # 对每个子段填充 Pedal_1 / Pedal_2
    for subseg_id, group in result.groupby('subseg_id'):
        orig = merged_all[merged_all['subseg_id'] == subseg_id]
        p1_list = orig[orig['Pedal_1'] != 0][['Pedal_1', 'Position_1', 'Duration_1']].values.tolist()
        #p2_list = orig[orig['Pedal_2'] != 0][['Pedal_2', 'Position_2', 'Duration_2']].values.tolist()

        idxs = group.index.tolist()
        for i, idx in enumerate(idxs):
            if i < len(p1_list):
                result.loc[idx, ['Pedal_1', 'Position_1', 'Duration_1']] = p1_list[i]
            else:
                result.loc[idx, ['Pedal_1', 'Position_1', 'Duration_1']] = [0, 0, 0]
            #if i < len(p2_list):
            #    result.loc[idx, ['Pedal_2', 'Position_2', 'Duration_2']] = p2_list[i]
            #else:
            #    result.loc[idx, ['Pedal_2', 'Position_2', 'Duration_2']] = [0, 0, 0]


    result = result.drop(columns=['Pedal', 'subseg_id', 'seg_id', 'row_idx', 'local_idx', 'content_idx'],
                         errors='ignore')
    return result.reset_index(drop=True)




def df_to_feature_vector_11dim(df):
    """11维特征向量转换"""
    # 确保所有列类型正确
    cols_to_fill = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration']
    for col in cols_to_fill:
        df[col] = df[col].fillna(0).astype(int)

    # 新增列已在前一步处理
    return df.values.T  # 返回 (11, n) 的矩阵


def process_sequences_with_sliding_window(input_data, max_length=256, stride=224):
    """
    对每条序列做滑动窗口切片，并生成对应的 mask。

    参数:
      - input_data: list of np.ndarray, 每个形状为 (channels, seq_len)
      - max_length: int, 窗口最大长度
      - stride: int, 窗口滑动步长 (stride < max_length 保证重叠)

    返回:
      - processed_data: np.ndarray, 形状 (num_windows, max_length, channels)
      - processed_masks: np.ndarray, 形状 (num_windows, max_length)
    """
    processed_chunks = []
    processed_masks = []

    for arr in input_data:
        # 确保类型
        arr = arr.astype(np.int32)
        channels, total_len = arr.shape

        # 从 0 开始，滑动切片
        start_idx = 0
        while start_idx < total_len:
            end_idx = start_idx + max_length
            # 如果到达尾部，不足一整窗时也要切出最后一段
            chunk = arr[:, start_idx:end_idx]
            chunk_len = chunk.shape[1]

            # 构造 mask：有效位 1，padding 位 0
            mask = np.ones(max_length, dtype=np.int32)

            if chunk_len < max_length:
                # padding 到 max_length
                pad_width = ((0, 0), (0, max_length - chunk_len))
                chunk = np.pad(chunk, pad_width, mode='constant', constant_values=0)

                # mask 的后半段置为 0
                mask[chunk_len:] = 0

            # 转置成 (max_length, channels)
            chunk = chunk.T

            processed_chunks.append(chunk)
            processed_masks.append(mask)

            # 下一步向前滑动 stride
            start_idx += stride

    # 合并所有窗口
    processed_data = np.stack(processed_chunks, axis=0)  # (num_windows, max_length, channels)
    processed_masks = np.stack(processed_masks, axis=0)  # (num_windows, max_length)

    return processed_data, processed_masks


def process_dataset(files, save_dir, dataset_type):
    """通用数据集处理函数"""
    feature_matrices = []
    processed_files = []  # 记录成功处理的文件路径

    # 处理数据文件
    for midi_path in files:
        tokens = tokenizer(midi_path)
        fields = extract_fields_from_tok_sequence(tokens)
        token_id_dict = build_token_id_dict(tokens, fields)
        processed_pairs = process_keys(token_id_dict)

        df = process_pairs(processed_pairs)

        # ===== 新增的 Pedal 处理流程 =====
        df = process_bar_column(df)
        df = preprocess_dataframe(df)
        df = process_pedal_special_values(df)
        # ==============================

        feature_matrix = df_to_feature_vector_11dim(df)
        #xlsx_path_ori = os.path.join("256", f"original.xlsx")
        #df.to_excel(xlsx_path_ori, index=False)
        #continue
        feature_matrices.append(feature_matrix)
        processed_files.append(midi_path)

    if not feature_matrices:
        print(f"{dataset_type} 数据集：没有有效数据需要保存")
        return

    # 准备Bar 数目。


    # 处理特征矩阵
    processed_data, processed_masks = process_sequences_with_sliding_window(feature_matrices)  # 修改点1：接收两个返回值

    # 保存路径配置
    data_filename = f"{dataset_type}_data.npy"
    mask_filename = f"{dataset_type}_mask.npy"  # 新增掩码文件名
    save_data_path = os.path.join(save_dir, data_filename)
    save_mask_path = os.path.join(save_dir, mask_filename)

    # 保存数据
    try:
        # 保存特征数据
        np.save(save_data_path, processed_data)
        print(f"\n{dataset_type} 特征数据保存成功！路径：{save_data_path}")
        print(f"特征数据形状：{processed_data.shape}")

        # 保存掩码数据
        np.save(save_mask_path, processed_masks)
        print(f"\n{dataset_type} 掩码数据保存成功！路径：{save_mask_path}")
        print(f"掩码数据形状：{processed_masks.shape}")

        # 元数据配置
        metadata = {
            "file_count": len(feature_matrices),
            "total_samples": processed_data.shape[0],
            "mask_samples": processed_masks.shape[0],  # 新增掩码样本数
            "max_length": 256,
            "feature_dims": 6,
            "processed_files": processed_files  # 使用实际成功记录
        }

        meta_path = os.path.join(save_dir, f"{dataset_type}_data.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        print(f"{dataset_type} 数据保存失败：{str(e)}")



# 主流程
if __name__ == "__main__":


    # 创建保存目录
    save_dir = "processed_data_small"
    os.makedirs(save_dir, exist_ok=True)
    # mask 也不能忘记，它是input
    process_dataset(train_files, save_dir, "train")
    process_dataset(val_files, save_dir, "val")
    process_dataset(test_files, save_dir, "test")

#D:\project\dataset\maestro-v3.0.0\preprocess.py:130: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`


