import json
from miditok import REMI
from miditok import TokenizerConfig

# =============================
# Create a standard REMI tokenizer
# =============================
config = TokenizerConfig(
    pitch_range=(21, 109),            # MIDI pitch range from A0 to C8
    beat_res={(0, 4): 96, (4, 20): 48},  # Temporal resolution (ticks per beat) for different bars
    num_velocities=64,                # Number of velocity bins
    use_sustain_pedals=True,          # Enable sustain pedal tokens
    sustain_pedal_duration=True,      # Track pedal duration
    num_tempos=64,                    # Number of tempo bins
    tempo_range=(40, 250),            # Tempo range in BPM
)
tokenizer = REMI(config)

# Save tokenizer configuration to file
tokenizer.save("./tokenizer.json")

# =============================
# Create a high-resolution REMI tokenizer
# =============================
config1 = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={
        (0, 4): 96,
        (4, 12): 48,
        (12, 36): 24,
        (36, 108): 12,
        (108, 120): 6,
    },  # Variable temporal resolutions for finer granularity in longer pieces
    num_velocities=64,
    use_sustain_pedals=True,
    sustain_pedal_duration=True,
    num_tempos=128,
    tempo_range=(40, 250),
)
tokenizer_hi = REMI(config1)

# Retrieve the vocabulary mapping
vocab = tokenizer_hi.vocab  # or use tokenizer_hi._vocab_base

# Save vocabulary as JSON for reference
with open("vocab_hi.json", "w") as f:
    json.dump(vocab, f, indent=2)

# Save the high-resolution tokenizer configuration
tokenizer_hi.save("./tokenizer_hi2.json")
