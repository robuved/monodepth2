from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy

import torch
import torch.utils.data as data
import random
from .kitti_dataset import KITTIRAWDataset
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class SequenceRawKittiDataset(data.IterableDataset):
    def __init__(self, data_path, video_paths, filenames, batch_size, img_ext, frame_idxs, shuffle=True, **kittirawdsargs):
        super().__init__()
        self.video_paths = video_paths
        self.batch_size = batch_size
        self.data_path = data_path
        self.video_ds = {}
        self.length = len(filenames)
        self.shuffle = shuffle
        for path in self.video_paths:
            video_filenames = []
            for fname in filenames:
                if fname.startswith(path):
                    video_filenames.append(fname)

            self.video_ds[path] = KITTIRAWDataset(data_path, video_filenames, frame_idxs=frame_idxs, img_ext=img_ext, **kittirawdsargs)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.video_paths)
        self.current_batch_ids = list(range(self.batch_size))
        self.next_video = self.batch_size
        self.data_loader = {}
        return self
    
    def __len__(self):
        return self.length

    def get_data_loader(self, path):
        if path not in self.data_loader:
            self.data_loader[path] = iter(DataLoader(self.video_ds[path], num_workers=0))

        return self.data_loader[path]
    
    def remove_data_loader(self, path):
        del self.data_loader[path]

    def __next__(self):
        data = []
        for idx, id in enumerate(self.current_batch_ids):
            if id >= len(self.video_paths):
                data.append(None)
                continue
            # id in list of videos

            # dataset indexed by path
            ds = self.get_data_loader(self.video_paths[id])
            
            while True:
                try:
                    d = next(ds)
                    break
                except StopIteration:
                    self.remove_data_loader(self.video_paths[id])
                    print(f"Completed iterating though {self.video_paths[id]}. Completed: {self.next_video}/{len(self.video_paths)} videos")
                    # finished ds
                    if self.next_video >= len(self.video_paths):
                        # no more videos
                        d = None
                        self.current_batch_ids[idx] = len(self.video_paths)
                        break

                    id = self.next_video
                    self.current_batch_ids[idx] = id
                    self.next_video += 1
                    ds = self.get_data_loader(self.video_paths[id])
            
            if d is None:
                data.append(None)
                continue
            data.append(d)

        not_none = list(filter(lambda x: x is not None, data))
        if not_none:
            batch_data = {}
            # concat data for each key
            for key in not_none[0].keys():
                placeholder = torch.zeros_like(not_none[0][key].squeeze(0))
                batch_key_data = [d[key].squeeze(0) if d is not None else placeholder for d in data]

                if isinstance(key, tuple) and key[0] == "imu":
                    batch_data[key] = pad_sequence(batch_key_data)
                else:
                    batch_data[key] = torch.stack(batch_key_data)
            return batch_data
        else:
            raise StopIteration    