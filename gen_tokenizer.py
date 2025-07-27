import json

from miditok import REMI
from pathlib import Path
from miditok import TokenizerConfig


config = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 96, (4, 20): 48},
    num_velocities=64,
    use_sustain_pedals=True,
    sustain_pedal_duration=True,
    num_tempos=64,
    tempo_range=(40, 250),
)
tokenizer = REMI(config)

tokenizer.save("./tokenizer.json")

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
tokenizer_hi = REMI(config1)
# 获取 vocab 映射表
vocab = tokenizer_hi.vocab  # 或 tokenizer_hi._vocab_base

# 保存为 JSON
with open("vocab_hi.json", "w") as f:
    json.dump(vocab, f, indent=2)
tokenizer_hi.save("./tokenizer_hi2.json")
