import pandas as pd
from miditok import REMI, TokenizerConfig, TokSequence
from miditoolkit import MidiFile
from symusic import Score


def process_music_df(raw_df):
    """
    完整的音乐数据处理流程：数据清洗转换
    参数改为直接接收DataFrame
    """
    required_columns = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']

    # 验证列是否存在
    missing_cols = [col for col in required_columns if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列：{', '.join(missing_cols)}")

    # 转换为标准数据格式
    df = raw_df[required_columns].copy()

    # 数据清洗
    df = df.replace(['', 'NULL', 'NA'], pd.NA)
    numeric_cols = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # 数据转换
    df['Bar'] = df['Bar'].fillna(0).astype(int)
    df['Position'] = df['Position'].ffill().astype('int32')
    df.loc[df['Pedal'] == 2882, 'Pitch'] = 93
    df.loc[df['Pedal'] == 1019, 'Pitch'] = 94
    df['Pitch'] = df['Pitch'].fillna(pd.NA)
    df['Velocity'] = df['Velocity'].fillna(0).astype(int)

    return df

def apply_business_rules(df):
    """执行所有新增业务规则"""
    df = df.replace(['', 'NULL', 'NA'], pd.NA)
    df = df.copy()

    numeric_cols = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    df['Bar'] = df['Bar'].replace(0, pd.NA)
    mask_314 = df['Pitch'] == 93
    mask_315 = df['Pitch'] == 94
    df.loc[mask_314, 'Pedal'] = 2882
    df.loc[mask_315, 'Pedal'] = 2883
    df['Pitch'] = df['Pitch'].where(~mask_314 & ~mask_315, pd.NA)
    df['Velocity'] = df['Velocity'].replace(0, pd.NA)

    position_dupe_mask = df['Position'] == df['Position'].shift(1)
    df.loc[position_dupe_mask, 'Position'] = pd.NA
    df['Position'] = df['Position'].astype('Int32')

    int_cols = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']
    df[int_cols] = df[int_cols].astype('Int32')

    return df

def transform_to_midi(df, midi_path):
    """将处理后的数据转换为MIDI文件"""
    ids = []
    for index, row in df.iterrows():
        for item in row:
            if pd.notna(item) and isinstance(item, (int, float)):
                item_int = int(item)
                # 检查当前数字是否为2882或2883且列表非空
                if (item_int == 2882 or item_int == 2883) and ids:
                    # 与上一个数字交换位置  这里的操作是因为生成pedal
                    prev = ids[-1]  # 获取上一个数字
                    ids[-1] = item_int  # 将上一个数字的位置设置为当前数字
                    ids.append(prev)  # 在列表末尾添加被替换的数字
                else:
                    ids.append(item_int)  # 否则直接添加数字

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

    decoded_music = tokenizer.decode([ids])
    decoded_music.dump_midi(midi_path)




def main(file_path):

    # 读取Excel文件 (移到主函数内)
    raw_df = pd.read_excel(file_path, engine='openpyxl')
    print("1")

    # 处理数据（传递DataFrame而非文件路径）
    processed_df = process_music_df(raw_df)
    processed_df = apply_business_rules(processed_df)

    # 转换为MIDI
    transform_to_midi(processed_df, "./output_cat_4.midi")
    print("处理完成，MIDI文件已生成。")


if __name__ == "__main__":
    main("../output/output_n1.xlsx")

