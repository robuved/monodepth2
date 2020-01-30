from options import MonodepthOptions
from pathlib import Path
import os
import random
import numpy as np
from utils import *


options = MonodepthOptions()
opts = options.parse()


def get_timestamps(path):
    lines = readlines(path)
    timestamps = [timestamp_to_datetime_float(line) for line in lines]
    indexes = list(range(len(timestamps)))
    ind = 0

    removed = 0
    while ind < len(timestamps) - 1:
        if timestamps[ind] <= timestamps[ind + 1]:
            ind += 1
            continue
        
        bad_index = ind
        start_back = bad_index
        upper_limit = timestamps[bad_index + 1]
        while timestamps[start_back] >= upper_limit and start_back >= 0:
            start_back -= 1
        count_to_remove_back = bad_index - start_back

        start_forward = bad_index + 1
        lower_limit = timestamps[bad_index]
        while timestamps[start_forward] <= lower_limit:
            start_forward += 1
        count_to_remove_forward = start_forward - bad_index

        if count_to_remove_back < count_to_remove_forward:
            timestamps = timestamps[:start_back + 1] + timestamps[bad_index + 1:]
            indexes = indexes[:start_back + 1] + indexes[bad_index + 1:]
            ind = start_back
            removed += bad_index - start_back
            
        else:
            timestamps = timestamps[:bad_index + 1] + timestamps[start_forward:]
            indexes = indexes[:bad_index + 1] + indexes[start_forward:]
            removed += start_back - bad_index - 1
    if removed:
        print(f"Warning: removing timestamps in {path}: {removed}/{len(lines)}")
    return indexes, timestamps

for file in Path(opts.data_path).glob("*/*/oxts/timestamps.txt"):
    indexes, timestamps = get_timestamps(file)
    output = file.with_name("unique_measurements.txt")
    np.array(indexes, dtype=np.int32).tofile(str(output))
