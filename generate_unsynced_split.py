from options import MonodepthOptions
from pathlib import Path
import os
import random
import numpy as np

options = MonodepthOptions()
opts = options.parse()

videos = []
for file in Path(opts.data_path).glob("*/*"):
    if file.is_dir():
        videos.append(file)

# print number of frames per video
extension = "png" if opts.png else "jpg"

count_videos = len(videos)
print("Number of videos", count_videos)

frame_counts = []
for path in videos:
    path_to_data = path / "image_00" / "data"
    number_frames = len(list(path_to_data.glob(f"*.{extension}")))
    print(path, number_frames)
    frame_counts.append(number_frames)

frame_counts = np.array(frame_counts)
print('Sorted lengths', np.sort(frame_counts))
# for full: 0 and small: 2
np.random.seed(0)

split = {
    "train": [],
    "val": [],
    "test": [],
}

for left_range, right_range, train_count, validation_count, test_count in  \
        [(0, 300, -1, 0, 0), (300, 1000, -1, 3, 3), (1000, 2600, -1, 2, 2), (2600, 10000, -1, 0, 0)]:
# for left_range, right_range, train_count, validation_count, test_count in  \
#         [(0, 300, 3, 1, 1), (300, 1000, 0, 0, 0), (1000, 4000, 0, 0, 0), (4000, 5000, 0, 0, 0), (5000, 10000, 0, 0, 0)]:
     
    to_pick_from = np.logical_and(frame_counts >= left_range, frame_counts < right_range)

    to_pick_indexes = np.where(to_pick_from)[0]
    to_pick_indexes = np.random.permutation(to_pick_indexes)
    print(f"From {left_range} to {right_range} ({(train_count, validation_count, test_count)} picks)")
    print(f"All: {to_pick_indexes}")

    if train_count < 0:
        train_count = len(to_pick_indexes) - validation_count - test_count

    train_indexes = to_pick_indexes[:train_count]
    validation_indexes = to_pick_indexes[train_count: train_count + validation_count]
    test_indexes = to_pick_indexes[train_count + validation_count:train_count + validation_count + test_count]

    split['train'].append(train_indexes)
    split['val'].append(validation_indexes)
    split['test'].append(test_indexes)


split = {k: np.concatenate(v) for k, v in split.items()}

splits_dir = os.path.join(os.path.dirname(__file__), "splits")
split_dir = Path(splits_dir) / "raw_odometry"
split_dir.mkdir(exist_ok=True)

rel_min, rel_max = min(opts.frame_ids), max(opts.frame_ids)

for key, video_ids in split.items():
    print(f"#######{key}########")
    print("IDS: ", video_ids)
    video_list_file_path = split_dir / f"{key}_video_list.txt"
    with open(video_list_file_path, "w") as out:
        for video_id in video_ids:
            path = videos[video_id]
            rel_path = path.relative_to(opts.data_path)
            out.write(str(rel_path) + "\n")
    print("Wrote video list for ", key)

    split_file = split_dir / f"{key}_files.txt"
    with open(split_file, "w") as out:
        for video_id in video_ids:
            path = videos[video_id]
            rel_path = path.relative_to(opts.data_path)
            path_to_data = path / "image_00" / "data"
            file_ids = [int(file.stem) for file in path_to_data.glob(f"*.{extension}")]
            file_ids = sorted(file_ids)
            max_ids = max(file_ids)
            for id in file_ids:
                if id + rel_min >= 0 and id + rel_max <= max_ids:
                    out.write(f"{str(rel_path)} {id} l\n{str(rel_path)} {id} r\n")
            print(f"{path}: {len(file_ids)}")
    print("Wrote files list for ", key)







