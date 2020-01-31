# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os

import argparse
import numpy as np
import PIL.Image as pil

from utils import readlines
from kitti_utils import generate_depth_map
from other_kitti_utils import load_oxts_packets_and_poses

# Source: https://github.com/utiasSTARS/pykitti

def export_gt_poses_kitti():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["raw_odometry"])
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    files = readlines(os.path.join(split_folder, "test_files.txt"))
    videos = readlines(os.path.join(split_folder, "test_video_list.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))

    for video in videos:
        oxts_paths = []
        for file in files:
            if file.startswith(video):
                folder, frame_id, _ = file.split()
                frame_id = int(frame_id)

                filepath_oxst = os.path.join(opt.data_path, folder,
                                             "oxts", "data", "{:010d}.txt".format(frame_id))

                oxts_paths.append(filepath_oxst)
        oxts = load_oxts_packets_and_poses(oxts_paths)
        poses_path = os.path.join(opt.data_path, video,
                                 "oxts", "poses.txt")
        poses = np.array([o[1] for o in oxts])
        print("Saving to {}".format(poses_path))

        np.savetxt(poses_path, poses)

if __name__ == "__main__":
    export_gt_poses_kitti()
