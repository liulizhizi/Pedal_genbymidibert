import numpy as np
from miditok import REMI
import json
from pathlib import Path


tokenizer_path = "../tokenizer_hi.json"
tokenizer = REMI(params=tokenizer_path)


paths_midis = list(Path("").glob("**/*.midi"))

# 对MIDI文件分词
tokens = tokenizer(paths_midis[0])


def time_scaling_augmentation(remi_tokens, ratio=1.0):
    new_remi = []
    current_bar, current_pos = 0, 0
    time_events = []

    # 提取绝对时间
    for token in tokens:
        if token == "Bar":
            current_bar = int(token.value)
        elif token == "Position":
            current_pos = int(token.value)
            abs_time = current_bar * 4 + current_pos
            time_events.append(abs_time)
        else:
            time_events.append(None)  # 占位符

    # 缩放时间并重新映射到Bar/Position
    scaled_times = [t * ratio if t is not None else None for t in time_events]
    new_remi = []
    bar_offset = 0
    last_bar = 0

    for i, token in enumerate(remi_tokens):
        if time_events[i] is not None:
            scaled_time = scaled_times[i]
            new_bar = int(scaled_time // 4)
            new_pos = int(round(scaled_time % 4))

            # 处理Bar变化
            if new_bar != last_bar:
                new_remi.append(f"Bar_{new_bar}")
                last_bar = new_bar
            new_remi.append(f"Position_{new_pos}")
        else:
            new_remi.append(token)  # 保留非时间令牌

    return new_remi


# 使用示例
scale_ratios = np.linspace(-0.25, 0.25, 11) + 1
augmented_data = []
for ratio in scale_ratios:
    augmented = time_scaling_augmentation(tokens, ratio)
    augmented_data.append(augmented)

midi_augmented = tokenizer.decode([augmented_data[10][0].ids])
midi_augmented.dump_midi("output/output_extend.midi")
print(tokens[0].ids)
print(augmented_data[10][0].ids)