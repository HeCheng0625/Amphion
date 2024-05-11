import os
import tqdm


if __name__ == "__main__":
    large_data_path = "/home/aiscuser/data/librilight/large_segs_15s"
    medium_data_path = "/home/aiscuser/data/librilight/medium_small_duplicate_15s"

    output_meta_file = "LibriLight/train_meta_file.txt"

    os.makedirs(os.path.dirname(output_meta_file), exist_ok=True)
    meta_files = []

    for root_path in [large_data_path, medium_data_path]:
        for speaker_id in tqdm.tqdm(os.listdir(root_path)):
            for book_id in os.listdir(os.path.join(root_path, speaker_id)):
                for seg_id in os.listdir(os.path.join(root_path, speaker_id, book_id)):
                    audio_path = os.path.join(root_path, speaker_id, book_id, seg_id)

                    if not audio_path.endswith(".flac"):
                        continue
                    meta_files.append(audio_path)

    with open(output_meta_file, "w") as f:
        for audio_path in tqdm.tqdm(meta_files):
            f.write(audio_path + "\n")
