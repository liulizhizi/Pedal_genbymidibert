import pandas as pd
from pandas import NA


def apply_business_rules(df):
    """执行所有新增业务规则"""
    # 阶段1：基础数据清洗
    df = df.replace(['', 'NULL', 'NA'], pd.NA)
    # 创建副本保证数据安全
    df = df.copy()

    # 转换数值类型
    numeric_cols = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    df['Pedal'] = pd.NA

    # 规则1：Bar列0值转NaN
    df['Bar'] = df['Bar'].replace(0, NA)

    # 规则2：Pitch与Pedal的关联处理
    mask_314 = df['Pitch'] == 93
    mask_315 = df['Pitch'] == 94
    df.loc[mask_314, 'Pedal'] = 1018
    df.loc[mask_315, 'Pedal'] = 1019
    df['Pitch'] = df['Pitch'].where(~mask_314 & ~mask_315, NA)

    # 规则3：Velocity列0值转NaN
    df['Velocity'] = df['Velocity'].replace(0, NA)

    # 生成位置重复掩码（与上一行相同）
    position_dupe_mask = df['Position'] == df['Position'].shift(1)

    # 保留第一个出现的值，后续重复值设为NaN
    df.loc[position_dupe_mask, 'Position'] = NA

    # 优化数据类型（处理可能产生的float类型）
    df['Position'] = df['Position'].astype('Int32')

    # 阶段4：最终类型优化
    int_cols = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']
    df[int_cols] = df[int_cols].astype('Int32')  # 支持NaN的整数类型

    return df


def transform_data(df):
    """执行完整的数据转换流程"""
    # 阶段1：基础数据清洗
    df = df.replace(['', 'NULL', 'NA'], pd.NA)

    # 转换数值类型
    numeric_cols = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # 阶段2：基础数据处理
    df['Bar'] = df['Bar'].fillna(0).astype('int16')
    df['Position'] = df['Position'].ffill().astype('int32')



    # 阶段4：最终类型优化
    int_cols = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']
    df[int_cols] = df[int_cols].astype('Int32')  # 支持NaN的整数类型

    return df


def process_music_data(raw_df, selection):
    """主处理流程（保持不变）"""
    required_columns = ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']

        # 列验证
    if missing := [col for col in required_columns if col not in raw_df.columns]:
        raise ValueError(f"缺少必要列：{', '.join(missing)}")
    if selection == 0:

        return transform_data(raw_df[required_columns].copy())
    else:

        return apply_business_rules(raw_df[required_columns].copy())




def has_positive_value(row):
    for col in ['Bar', 'Position', 'Pitch', 'Velocity', 'Duration', 'Pedal']:
        try:
            if row[col] > 0:
                return True
        except TypeError:
            # 处理NA类型
            continue
    return False


# 使用示例
if __name__ == "__main__":
    try:
        try:
            raw_df = pd.read_excel("../output_cat.xlsx", engine='openpyxl')
        except FileNotFoundError:
            raise FileNotFoundError(f"文件不存在：../output_cat.xlsx")
        except Exception as e:
            raise RuntimeError(f"处理失败：{str(e)}")
        df = process_music_data(raw_df,1)
        print("处理结果样例：")
        print(df.head(3))
        print("\n数据类型：")
        print(df.dtypes)


        # 阶段3：应用业务规则
        valid_rows = pd.Series([False] * len(df))  # 初始化所有行为False

        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            if has_positive_value(row):
                # 找到第一个有效行，标记该行及其之前的所有行
                valid_rows[:i + 1] = True
                break

        df = df[valid_rows]

        df.to_excel("output_cat_1.xlsx", index=False)
        print("\n处理结果已保存到 output_cat_1.xlsx")

    except Exception as e:
        print(f"处理错误：{str(e)}")