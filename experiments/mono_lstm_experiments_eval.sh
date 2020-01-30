# Depth evaluation

# Standard model with rescaling
python ../evaluate_depth.py --data_path ../kitti_data_unsynced \
  --model_name M_640x192_no_IMU \
  --load_weights_folder ../tensorboard/M_640x192_no_IMU/models/weight_X/ \
  --split raw_odometry

# Standard model without rescaling
python ../evaluate_depth.py --data_path ../kitti_data_unsynced \
  --model_name M_640x192_no_IMU \
  --load_weights_folder ../tensorboard/M_640x192_no_IMU/models/weight_X/ \
  --split raw_odometry \
  --disable_median_scaling

# Pose2 Without Fuse with rescaling
python ../evaluate_depth.py --data_path ../kitti_data_unsynced \
  --model_name M_640x192_IMU_2p \
  --load_weights_folder ../tensorboard/M_640x192_IMU_2p/models/weight_X/ \
  --split raw_odometry

# Pose2 Without Fuse without rescaling
python ../evaluate_depth.py --data_path ../kitti_data_unsynced \
  --model_name M_640x192_IMU_2p \
  --load_weights_folder ../tensorboard/M_640x192_IMU_2p/models/weight_X/ \
  --split raw_odometry \
  --disable_median_scaling

# Pose2 With Fuse with rescaling
python ../evaluate_depth.py --data_path ../kitti_data_unsynced \
  --model_name M_640x192_IMU_2p_pose_fuse \
  --load_weights_folder ../tensorboard/M_640x192_IMU_2p_pose_fuse/models/weight_X/ \
  --split raw_odometry

# Pose2 With Fuse without rescaling
python ../evaluate_depth.py --data_path ../kitti_data_unsynced \
  --model_name M_640x192_IMU_2p_pose_fuse \
  --load_weights_folder ../tensorboard/M_640x192_IMU_2p_pose_fuse/models/weight_X/ \
  --split raw_odometry \
  --disable_median_scaling
