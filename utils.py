# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib
import datetime
import glob
from pathlib import Path
import numpy as np


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def timestamp_to_datetime_float(timestamp):
    return datetime.datetime.strptime(timestamp[:-3], '%Y-%m-%d %H:%M:%S.%f').timestamp()


def get_timestamps(path):
    lines = readlines(path)
    timestamps = [timestamp_to_datetime_float(line) for line in lines]
    
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
            # to_remove.append((start_back + 1, bad_index))
            # ind += 1

            timestamps = timestamps[:start_back + 1] + timestamps[bad_index + 1:]
            ind = start_back
            removed += bad_index - start_back
            
        else:
            # to_remove.append((bad_index + 1, start_forward - 1))
            # ind = start_forward

            timestamps = timestamps[:bad_index + 1] + timestamps[start_forward:]
            removed += start_back - bad_index - 1
    # if len(to_remove):
    #     print(f"Warning: removing timestamps in {path}")
    #     total_removed = 0
    #     for start, end in reversed(to_remove):
    #         count_removed = end - start + 1
    #         total_removed += count_removed
    #         print(f"\t {count_removed}: {start} to {end}; {lines[start]} to {lines[end]}; {timestamps[start]} to {timestamps[end]}")
    #         timestamps = timestamps[:start] + timestamps[end + 1:]
    #         lines = lines[:start] + lines[end + 1:]
    #     print(f"\t\tRemoved: {total_removed}/{len(lines)}")
    if removed:
        print(f"Warning: removing timestamps in {path}: {removed}/{len(lines)}")

    timestamps = np.array(timestamps)   
    for i in range(len(timestamps) - 1):
        if timestamps[i] > timestamps[i + 1]:
             print(timestamps[i], timestamps[i + 1])
    good = timestamps[:-1] <= timestamps[1:]
    bad = np.logical_not(good)
    bad_index = np.where(bad)[0]
    if len(bad_index):
        print("bad", path, bad_index, lines[bad_index[0]], lines[bad_index[0] + 1], timestamps[bad_index[0]], timestamps[bad_index[0] + 1])
    assert(np.all(good))
    return timestamps


def get_imu_data(path):
    files = glob.glob(os.path.join(path, "*.txt"))
    data = {}
    for file in files:
        contents = readlines(file)[0].split(" ")
        accelerations = contents[11:14]
        ang_velocity = contents[17:20]
        accelerations = [float(a) for a in accelerations]
        ang_velocity = [float(a) for a in ang_velocity]

        index = int(Path(file).stem)
        data[index] = (accelerations, ang_velocity)
    
    acc_list = []
    ang_vel_list = []
    for i in range(len(data.items())):
        assert(i in data.keys())
        acc_list.append(data[i][0])
        ang_vel_list.append(data[i][1])
    return acc_list, ang_vel_list


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))
