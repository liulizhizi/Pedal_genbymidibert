import pandas as pd
import numpy as np


def restore_custom_format(df):
    df = df.copy()
    df['seg_id'] = df['Bar'].ne(0).cumsum()

    restored_rows = []

    for seg_id, group in df.groupby('seg_id'):
        # 跳过 padding 空白行：Bar==0 且 Pitch==0 且 Pedal_1/Pedal_2==0
        #is_padding = (
        #    (group['Bar'] == 0) &
        #    (group['Pitch'] == 0) &
        #    (group['Pedal_1'] == 0) &
        #    (group['Pedal_2'] == 0)
        #)
        is_padding = (
                (group['Bar'] == 0) &
                (group['Pitch'] == 0) &
                (group['Pedal_1'] == 0)
        )
        group = group[~is_padding].copy()

        # 找出段首非零 Bar 的行（保留段首）
        bar_row = group[group['Bar'] != 0].head(1).copy()
        bar_row = bar_row[['Bar', 'Position', 'Pitch', 'Velocity', 'Duration']].copy()
        bar_row['Pedal'] = 0
        restored_rows.append(bar_row)

        # pedal_1 事件
        pedal1_rows = group[group['Pedal_1'] != 0][['Position_1', 'Duration_1', 'Pedal_1']].copy()
        pedal1_rows = pedal1_rows.rename(columns={
            'Position_1': 'Position',
            'Duration_1': 'Duration',
            'Pedal_1': 'Pedal'
        })
        pedal1_rows['Pitch'] = 0
        pedal1_rows['Velocity'] = 0
        pedal1_rows['Bar'] = 0

        # pedal_2 事件
        #pedal2_rows = group[group['Pedal_2'] != 0][['Position_2', 'Duration_2', 'Pedal_2']].copy()
        #pedal2_rows = pedal2_rows.rename(columns={
        #    'Position_2': 'Position',
        #    'Duration_2': 'Duration',
        #    'Pedal_2': 'Pedal'
        #})
        #pedal2_rows['Pitch'] = 0
        #pedal2_rows['Velocity'] = 0
        #pedal2_rows['Bar'] = 0

        # pitch 行
        pitch_rows = group[group['Pitch'] != 0][['Bar', 'Position', 'Pitch', 'Velocity', 'Duration']].copy()
        pitch_rows['Pedal'] = 0

        # 合并所有事件行
        #merged = pd.concat([pedal1_rows, pedal2_rows, pitch_rows], ignore_index=True)
        merged = pd.concat([pedal1_rows, pitch_rows], ignore_index=True)
        merged = merged[['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']]
        merged = merged.sort_values(by='Position').reset_index(drop=True)

        restored_rows.append(merged)

    final_df = pd.concat(restored_rows, ignore_index=True)
    return final_df



def process_bar_events(df):
    """
    处理Bar非零事件：当Bar非零时，将4填入下一行Bar列，并移除当前行

    参数:
    df (pd.DataFrame): 添加踏板事件后的DataFrame

    返回:
    pd.DataFrame: 处理Bar事件后的DataFrame
    """
    # 存储需要修改的行索引及值 [索引, 新值]
    bar_updates = []
    # 存储需要移除的行索引
    rows_to_remove = []

    # 遍历DataFrame处理Bar事件
    for i in range(len(df)):
        # 只处理原始行（非踏板事件行）
        if 'Source' in df.columns and df.at[i, 'Source'] == 'pedal':
            continue

        if df.at[i, 'Bar'] != 0:
            # 查找下一行索引
            next_idx = i + 1
            if next_idx < len(df):
                # 记录需要更新的行
                bar_updates.append((next_idx, 4))

            # 记录需要移除的行
            rows_to_remove.append(i)

    # 应用Bar更新
    for idx, new_value in bar_updates:
        df.at[idx, 'Bar'] = new_value

    # 移除标记的行
    if rows_to_remove:
        # 使用索引位置而不是索引标签避免问题
        df = df.drop(index=rows_to_remove)

    return df


def clean_data(df):
    """
    最后一步数据清理：
    1. 如果Position值与上一行相同，改为空值
    2. 将数值0全部替换为空值

    参数:
    df (pd.DataFrame): 处理完Bar事件后的DataFrame

    返回:
    pd.DataFrame: 清理后的最终DataFrame
    """
    # 1. 处理Position相同值 - 使用安全的迭代方式
    if 'Position' in df.columns:
        for i in range(1, len(df)):
            # 确保索引存在
            if pd.notna(df.at[i, 'Position']) and pd.notna(df.at[i - 1, 'Position']):
                if df.at[i, 'Position'] == df.at[i - 1, 'Position']:
                    df.at[i, 'Position'] = np.nan

    # 2. 将所有数值0替换为空值
    # 指定需要处理的列
    columns_to_clean = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']

    for col in columns_to_clean:
        if col in df.columns:
            # 使用mask方法安全替换0值
            df[col] = df[col].mask(df[col] == 0, np.nan)

    return df


def process_midi_df(df):
    """
    完整的MIDI数据处理流程

    参数:
    df (pd.DataFrame): 原始数据DataFrame

    返回:
    pd.DataFrame: 处理后的DataFrame
    """
    # 步骤1: 添加踏板事件行
    df = restore_custom_format(df)

    # 步骤2: 处理Bar事件
    df = process_bar_events(df)

    # 重置索引确保连续性 (修复问题的关键步骤)
    df.reset_index(drop=True, inplace=True)

    # 步骤3: 数据清理
    df = clean_data(df)

    # 删除临时列
    df = df.drop(columns=['Pedal_1', 'Position_1', 'Duration_1', 'Source'], errors='ignore')

    return df


# 示例用法
if __name__ == "__main__":
    # 读取输入文件
    input_file = 'output_n.xlsx'
    df = pd.read_excel(input_file)

    # 处理数据
    processed_df = process_midi_df(df)

    # 保存结果
    output_file = 'output_nn.xlsx'
    processed_df.to_excel(output_file, index=False)
    print(f"处理完成! 输入: {input_file}, 输出: {output_file}")
