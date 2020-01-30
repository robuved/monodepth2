from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy

import torch
import torch.utils.data as data
import random
from kitti_dataset import KITTIRAWDataset
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class SequenceRawKittiDataset(data.IterableDataset):
    def __init__(self, data_path, video_paths, batch_size, img_ext, frame_idxs, **kittirawdsargs):
        super().__init__()
        self.video_paths = video_paths
        self.batch_size = batch_size
        self.data_path = data_path
        self.video_ds = {}
        self.steps = 0
        for path in self.video_paths:
            path_to_data = Path(data_path) / path /  "image_00" / "data"
            file_ids = sorted([int(file.stem) for file in path_to_data.glob(f"*.{img_ext}")])
            sorted_frame_idxs = sorted(frame_idxs)

            file_ids_to_process = []
            for id in file_ids:
                if id + sorted_frame_idxs[0] >= 0 and id + sorted_frame_idxs[-1] < file_ids[-1] and \
                        (not file_ids_to_process or id + sorted_frame_idxs[0] > file_ids_to_process[-1] + file_ids[-1]):
                    file_ids_to_process.append(id)

            print(file_ids_to_process[:20])  
            filepaths = [f"{path} {id} l" for id in file_ids_to_process]
            self.video_ds[path] = KITTIRAWDataset(data_path, filepaths, img_ext=img_ext, **kittirawdsargs)
            self.steps += len(self.video_ds[path])

    def __iter__(self):
        random.shuffle(self.video_paths)
        self.current_batch_ids = list(range(self.batch_size))
        self.next_pos_in_seq = [0 for id in self.current_batch_ids]
        self.next_video = self.batch_size
        self.data_loader = {}

    def get_data_loader(self, path):
        if path not in self.data_loader:
            self.data_loader[path] = DataLoader(self.video_ds[path], num_workers=2)

        return self.data_loader[path]

    def __next__(self):
        data = []
        for idx in range(self.batch_size):
            # id in list of videos
            id = self.current_batch_ids[idx]

            # dataset indexed by path
            ds = self.get_data_loader(self.video_paths[id])

            if self.next_pos_in_seq[idx] >= len(ds):
                # finished ds
                if self.next_video >= len(self.video_paths):
                    # no more videos
                    continue

                id = self.next_video
                self.current_batch_ids[idx] = id
                self.next_video += 1
                self.next_pos_in_seq[idx] = 0
                ds = self.get_data_loader(self.video_paths[id])

            data.append(ds[self.next_pos_in_seq[idx]])
            self.next_pos_in_seq[idx] += 1 
        
        if data:
            batch_data = {}
            # concat data for each key
            for key in data[0].keys():
                batch_key_data = [d[key] for d in data]
                if isinstance(key, tuple) and key[0] == "imu":
                    batch_data[key] = pad_sequence(batch_key_data)
                else:
                    batch_data[key] = torch.stack(batch_key_data)
        else:
            raise StopIteration    