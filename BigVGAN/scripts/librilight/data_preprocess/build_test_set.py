import os
import random


if __name__ == "__main__":
    output_valid_meta_file = "LibriLight/valid_meta_file.txt"  # 120 cases
    output_test_meta_file = "LibriLight/test_meta_file.txt"

    os.makedirs(os.path.dirname(output_valid_meta_file), exist_ok=True)

    test_folder = "/home/aiscuser/data/librispeech/LibriSpeech/test-clean"

    audio_paths = []

    for speaker_id in os.listdir(test_folder):
        for book_id in os.listdir(os.path.join(test_folder, speaker_id)):
            for seg_id in os.listdir(os.path.join(test_folder, speaker_id, book_id)):
                audio_path = os.path.join(test_folder, speaker_id, book_id, seg_id)

                if not audio_path.endswith(".flac"):
                    continue

                audio_paths.append(audio_path)

    random.shuffle(audio_paths)

    with open(output_valid_meta_file, "w") as f:
        for audio_path in audio_paths[:120]:
            f.write(audio_path + "\n")

    with open(output_test_meta_file, "w") as f:
        for audio_path in audio_paths[120:]:
            f.write(audio_path + "\n")
