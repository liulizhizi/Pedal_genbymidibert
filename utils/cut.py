import mido

def crop_midi(input_file, output_file, start_time, end_time):
    mid = mido.MidiFile(input_file)
    new_mid = mido.MidiFile()

    for track in mid.tracks:
        new_track = mido.MidiTrack()
        time_counter = 0  # 记录时间
        for msg in track:
            time_counter += msg.time  # 计算时间
            if start_time <= time_counter <= end_time:
                new_track.append(msg)
        new_mid.tracks.append(new_track)

    new_mid.save(output_file)
    print(f"裁剪完成，保存至 {output_file}")

# 示例：裁剪 5 秒到 15 秒的部分
crop_midi("../2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi", "2000/output.midi", 5 * 480, 15 * 480)
