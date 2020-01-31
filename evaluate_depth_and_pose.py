#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from layers import disp_to_depth

from layers import transformation_from_parameters
from utils import readlines
from options import MonodepthOptions
from datasets import SequenceRawKittiDataset, KITTIOdomDataset, KITTIRAWDataset
import networks
import torch.nn as nn
from layers import transformation_from_matrix, rot_from_axisangle, rot_translation_from_transformation
from collections import defaultdict
import cv2
from matplotlib import pyplot as plt


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o, do_scaling=True):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    if do_scaling:
        scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    else:
        scale = 1
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def evaluate(opt):
    """Evaluate odometry on the KITTI dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    # Depth
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    # Pose
    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

    pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))

    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))

    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()

    if opt.use_imu:
        imu_lstm = nn.LSTM(6, opt.lstm_hidden_size, opt.lstm_num_layers)
        imu_lstm.cuda()
        imu_lstm.eval()
        lstm_hs = None

        hidden_to_imu = torch.nn.Sequential(
            torch.nn.Linear(opt.lstm_hidden_size, 6),
        )
        hidden_to_imu.cuda()
        hidden_to_imu.eval()

        if opt.pose_fuse:
            pose_fuse_mlp = torch.nn.Sequential(
                torch.nn.Linear(24, opt.pose_mlp_hidden_size),
                torch.nn.Sigmoid(),
                torch.nn.Linear(opt.pose_mlp_hidden_size, 6),
            )
            pose_fuse_mlp.cuda()
            pose_fuse_mlp.eval()

    img_ext = '.png' if opt.png else '.jpg'

    pred_disps = []
    scale_factors = []
    
    kitty_odom = False
    if opt.eval_split.startswith("odom"):
        kitty_odom = True

    if kitty_odom:
        ids = [int(opt.eval_split.split("_")[1])]
    else:
        splits_dir = os.path.join(os.path.dirname(__file__), "splits")
        videonames = readlines(os.path.join(splits_dir, opt.eval_split, "test_video_list.txt"))
        ids = videonames

    for videoname in ids:
        if kitty_odom:
            filenames = readlines(
                     os.path.join(splits_dir, opt.eval_split,
                     "test_files_{:02d}.txt".format(videoname)))
        else:
            filenames = readlines(os.path.join(splits_dir, opt.eval_split,"test_files.txt"))
        if kitty_odom:

            dataset = KITTIOdomDataset(opt.data_path, filenames, opt.height, opt.width,
                               [0, 1], 4, is_train=False, use_imu=False)
            dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)
        else:
            if opt.use_imu:
                dataset = SequenceRawKittiDataset(opt.data_path, [videoname], filenames, 1, 
                                           imu_data_path=opt.imu_data_path,
                                           img_ext=img_ext, frame_idxs=[0, 1],
                                           height=encoder_dict['height'], width=encoder_dict['width'],
                                           num_scales=4, is_train=False)
                dataloader = DataLoader(dataset, shuffle=False, num_workers=0)
            else:
                filenames = list(filter(lambda f: f.startswith(videoname), filenames))
                dataset = KITTIRAWDataset(opt.data_path, filenames, opt.height, opt.width,
                               [0, 1], 4, is_train=False, use_imu=False)
                dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)
        # pred_poses = [np.eye(4).reshape(1, 4, 4)]
        pred_poses = []
        imu_scale_factors = []
        
        print("EVALUATING ", opt.model_name)

        print("-> Computing pose predictions")

        opt.frame_ids = [0, 1]  # pose network only takes two frames as input

        with torch.no_grad():
            for inputs in dataloader:
                for key, ipt in inputs.items():
                    inputs[key] = ipt.cuda()
                    if opt.use_imu:
                        inputs[key] = inputs[key].squeeze(0)
                input_color = inputs[("color", 0, 0)]
                feature = encoder(input_color)
                output = depth_decoder(feature)

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                pred_disps.append(pred_disp)

                all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in sorted(opt.frame_ids)], 1)

                features = [pose_encoder(all_color_aug)]
                axisangle, translation = pose_decoder(features)
                outputs = {}
                outputs[("cam_T_cam", 0, 1)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=False)

                T = outputs[("cam_T_cam", 0, 1)]
                if opt.use_imu:
                    outputs = predict_poses_from_imu2(opt, inputs, imu_lstm, lstm_hs, hidden_to_imu)
                    T_better = outputs[("cam_T_cam_imu", 0, 1)]
                    if opt.pose_fuse:
                        fuse_poses(opt, outputs, pose_fuse_mlp)
                        T_better = outputs[("cam_T_cam_fuse", 0, 1)]

                    R, t = rot_translation_from_transformation(T)
                    Rb, tb = rot_translation_from_transformation(T_better)
                    imu_scale_factor = torch.sum(tb * t) / torch.sum(t ** 2)

                    imu_scale_factors.append(imu_scale_factor.cpu().numpy())
                    # scale_factors.append(imu_scale_factors)

                    T = T_better

                pred_poses.append(T.cpu().numpy())
            
            pred_poses = np.concatenate(pred_poses)
        
            if opt.eval_split.startswith("odom"):
                gt_poses_path = os.path.join(opt.data_path, "poses", "{:02d}.txt".format(videoname))
            else:
                gt_poses_path = os.path.join(opt.data_path, videoname, "oxts", "poses.txt")
             
            eval_pose(opt, pred_poses, gt_poses_path)
        scale_factors = {}
        if imu_scale_factors:
            scale_factors["IMU factor"] = imu_scale_factors
    pred_disps = np.concatenate(pred_disps)
    if not kitty_odom:
        eval_depth(opt, pred_disps, scale_factors)


def eval_pose(opt, pred_poses, gt_poses_path):

    gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    gt_global_poses = np.concatenate(
        (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]

    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(
            np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))
        # gt_local_poses.append(
        #     np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i]))

    ates = []
    ates_no_scaling = []
    num_frames = gt_xyzs.shape[0]
    track_length = 5
    # print(pred_poses.shape, gt_global_poses.shape)
    for i in range(0, num_frames - 1):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))
        ates_no_scaling.append(compute_ate(gt_local_xyzs, local_xyzs, do_scaling=False))

    print("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))
    print("\n   Trajectory error(no scaling): {:0.3f}, std: {:0.3f}\n".format(np.mean(ates_no_scaling), np.std(ates_no_scaling)))

    save_path = os.path.join(opt.load_weights_folder, "poses.npy")
    np.save(save_path, pred_poses)
    print("-> Predictions saved to", save_path)
    
    
    # print("Plotting ...")
    # def compose(poses):
    #     prev_pose = np.eye(4)
    #     global_poses = [prev_pose]

    #     for pose in poses:
    #         global_poses.append(global_poses[-1] @ pose)
    #     return global_poses

    # def collect_positions(poses):
    #     return [p[:3, 3] for p in poses]

    # gt_positions = collect_positions(compose(gt_local_poses))
    # predicted_positions = collect_positions(compose(pred_poses))

    # gt_positions = np.array(gt_positions)[:, :2]
    # predicted_positions = np.array(predicted_positions)[:, :2]
    # plt.figure()
    # plt.plot(gt_positions[:, 0], gt_positions[:, 1], label='gt')
    # plt.title(gt_poses_path)
    # plt.legend()

    # plt.figure()
    # plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], label='pred')
    # plt.title(gt_poses_path)
    # plt.legend()
    # plt.show()


def eval_depth(opt, pred_disps, scaling_factors):
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")

    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")
    print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Median scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    for k, factors in scaling_factors.items():
        factors = np.array(factors)
        print(factors, factors.shape)
        med = np.median(factors)
        print("{} scaling ratios | med: {:0.3f} | std: {:0.3f}".format(k, med, np.std(factors / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


def predict_poses_from_imu2(opt, inputs, imu_lstm, lstm_hs, hidden_to_imu):
    # get relative poses ordered
    sorted_frame_ids = sorted(opt.frame_ids)

    # propagate IMU data though LSTM and mapping linear layer + add input
    imu_timestamps = inputs[("imu", "timestamps")]
    imu_measurements = inputs[("imu", "measurements")]
    imu_features, lstm_hs = imu_lstm(imu_measurements, lstm_hs)
    imu_corrected = imu_measurements + hidden_to_imu(imu_features)
    # BATCH size X SEQUENCE length X 6
    imu_index = 0

    cam_T_cam = defaultdict(lambda: [])

    b_size = inputs[("timestamp", 0)].shape[0]

    for b_slice in range(b_size):
        imu_index = 0
        for f_id1, f_id2 in zip(sorted_frame_ids[:-1], sorted_frame_ids[1:]):
            ts1 = inputs[("timestamp", f_id1)][b_slice]
            ts2 = inputs[("timestamp", f_id2)][b_slice]

            cum_rotation = torch.eye(3).repeat(1, 1, 1).cuda()
            cum_pos = torch.zeros(3).cuda()
            cum_vel = torch.zeros(3).cuda()

            while imu_index < len(imu_timestamps):
                ts = imu_timestamps[imu_index, b_slice, 0]
                if ts < 0 or ts > ts1:
                    break
                prev_ts = ts1
                if imu_index - 1 >= 0:
                    prev_ts = torch.max(prev_ts, imu_timestamps[imu_index - 1, b_slice, 0])

                current_ts = ts
                if imu_index + 1 < len(imu_timestamps):
                    next_ts = imu_timestamps[imu_index + 1, b_slice, 0]
                    if next_ts > 0 and next_ts > ts2:
                        current_ts = ts2

                dt = current_ts - prev_ts

                c_gyro = imu_corrected[imu_index, b_slice, 3:6] * dt

                rotations = rot_from_axisangle(c_gyro[None, None, :])[:, :3, :3]

                cum_rotation = torch.matmul(cum_rotation, rotations)

                c_acc = imu_corrected[imu_index, b_slice, :3]

                global_acc = torch.matmul(cum_rotation, c_acc[:, None])

                cum_vel = cum_vel + global_acc[0, 0, :] * dt
                cum_pos = cum_pos + cum_vel * dt + 0.5 * global_acc[0, 0, :] * dt ** 2

                imu_index += 1

            cam_T_cam[(f_id1, f_id2)].append((cum_rotation, cum_pos[None, :, None]))

    for k, v in cam_T_cam.items():
        rots, pos = zip(*v)
        rots = torch.cat(rots)
        pos = torch.cat(pos)
        cam_T_cam[k] = (rots, pos)
    # compose IMU to get relative poses
    outputs = {}
    for f_id in opt.frame_ids[1:]:

        to_compose = range(0, f_id) if f_id > 0 else range(f_id, 0)
        reverse = f_id < 0

        T = None
        for f_id_int in to_compose:
            c_T = transformation_from_matrix(*cam_T_cam[(f_id_int, f_id_int + 1)], invert=reverse)
            if T is None:
                T = c_T
            else:
                T = torch.matmul(T, c_T) if not reverse else torch.matmul(c_T, T)
        outputs[("cam_T_cam_imu", 0, f_id)] = T

    return outputs


def fuse_poses(opt, outputs, pose_fuse_mlp):
    def transformation_to_tensor(tr_batch):
        R, t = rot_translation_from_transformation(tr_batch)
        return torch.cat([R.reshape(-1, 9), t.reshape(-1, 3)], dim=1)

    for f_id in opt.frame_ids[1:]:
        pose_net = transformation_to_tensor(outputs[("cam_T_cam", 0, f_id)])
        pose_imu = transformation_to_tensor(outputs[("cam_T_cam_imu", 0, f_id)])
        pose_fuse_input = torch.cat([pose_net, pose_imu], dim=1)
        pose_fuse_output = pose_fuse_mlp(pose_fuse_input)
        axisangle = pose_fuse_output[:, :3].reshape(-1, 1, 3)
        tr = pose_fuse_output[:, 3:6].reshape(-1, 1, 3)
        T = transformation_from_parameters(axisangle, tr)
        outputs[("cam_T_cam_fuse", 0, f_id)] = T


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())



'''
EVALUATING  M_640x192_author

   Trajectory error: 1.687, std: 0.540
   Trajectory error(no scaling): 1.689, std: 0.540
   Trajectory error: 1.986, std: 0.071
   Trajectory error(no scaling): 1.988, std: 0.071
   Trajectory error: 0.432, std: 0.162
   Trajectory error(no scaling): 0.433, std: 0.162
   Trajectory error: 1.222, std: 0.325
   Trajectory error(no scaling): 1.223, std: 0.325
   Trajectory error: 1.175, std: 0.286
   Trajectory error(no scaling): 1.176, std: 0.286
 Median scaling ratios | med: 30.626 | std: 0.142

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.123  &   1.513  &   4.821  &   0.208  &   0.883  &   0.951  &   0.975  \\

EVALUATING  M_640x192_no_IMU
   Trajectory error: 1.222, std: 0.385
   Trajectory error(no scaling): 1.684, std: 0.539
   Trajectory error: 1.429, std: 0.054
   Trajectory error(no scaling): 1.983, std: 0.071
   Trajectory error: 0.315, std: 0.121
   Trajectory error(no scaling): 0.430, std: 0.161
   Trajectory error: 0.890, std: 0.241
   Trajectory error(no scaling): 1.218, std: 0.325
   Trajectory error: 0.861, std: 0.208
   Trajectory error(no scaling): 1.172, std: 0.286
 Median scaling ratios | med: 61.045 | std: 0.191

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.456  &   4.760  &  11.863  &   0.596  &   0.296  &   0.547  &   0.755  \\

EVALUATING  M_640x192_IMU_2p_pose_fuse
   Trajectory error: 1.409, std: 0.462
   Trajectory error(no scaling): 1.688, std: 0.539
   Trajectory error: 1.662, std: 0.094
   Trajectory error(no scaling): 1.986, std: 0.071
   Trajectory error: 0.356, std: 0.131
   Trajectory error(no scaling): 0.432, std: 0.162
   Trajectory error: 1.011, std: 0.265
   Trajectory error(no scaling): 1.222, std: 0.325
   Trajectory error: 0.969, std: 0.241
   Trajectory error(no scaling): 1.175, std: 0.286
IMU factor scaling ratios | med: -0.000 | std: 126.959
 Median scaling ratios | med: 63.633 | std: 0.194

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.455  &   4.751  &  11.872  &   0.597  &   0.296  &   0.548  &   0.756  \\

'''