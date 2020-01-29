from options import MonodepthOptions
from pathlib import Path
import os
import random

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

for path in videos:
    path_to_data = path / "image_00" / "data"
    number_frames = len(list(path_to_data.glob(f"*.{extension}")))
    print(path, number_frames)
random.seed(0)

random.shuffle(videos)

validation_count = int((0.15) * count_videos)
test_count = int((0.15) * count_videos)
train_count = count_videos - validation_count - test_count
print(train_count, validation_count, test_count)

split = {
    "train": videos[:train_count],
    "validation": videos[train_count:train_count + validation_count],
    "test": videos[train_count + validation_count:],
}

splits_dir = os.path.join(os.path.dirname(__file__), "splits")
split_dir = Path(splits_dir) / "raw_odometry"
split_dir.mkdir(exist_ok=True)

rel_min, rel_max = min(opts.frame_ids), max(opts.frame_ids)

for key, video_paths in split.items():
    video_list_file_path = split_dir / f"{key}_video_list.txt"
    with open(video_list_file_path, "w") as out:
        for path in video_paths:
            rel_path = path.relative_to(opts.data_path)
            out.write(str(rel_path) + "\n")
    print("Wrote video list for ", key)

    split_file = split_dir / f"{key}_files.txt"
    with open(split_file, "w") as out:
        for path in video_paths:
            rel_path = path.relative_to(opts.data_path)
            path_to_data = path / "image_00" / "data"
            file_ids = [int(file.stem) for file in path_to_data.glob(f"*.{extension}")]
            file_ids = sorted(file_ids)
            max_ids = max(file_ids)
            for id in file_ids:
                if id + rel_min >= 0 and id + rel_max <= max_ids:
                    out.write(f"{str(rel_path)} {id} l\n{str(rel_path)} {id} r\n")
    print("Wrote files list for ", key)







