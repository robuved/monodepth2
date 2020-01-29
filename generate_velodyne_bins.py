from options import MonodepthOptions
from pathlib import Path
from utils import readlines
import numpy as np

options = MonodepthOptions()
opts = options.parse()


for file in Path(opts.data_path).glob("*/*/velodyne_points/data/*.txt"):
    lines = readlines(file)
    data = []
    for line in lines:
        fragments = line.split(" ")
        numbers = [float(i) for i in fragments[:4]]
        data.append(numbers)
    data = np.array(data, dtype=np.float32).reshape(-1)
    bin_path = str(file.with_suffix(".bin"))
    data.tofile(str(file.with_suffix(".bin")))
    print('Saved data to ', bin_path)