# Depth evaluation

# aathor trained
python ../evaluate_pose.py \
  --data_path ../kitti_odom \
  --model_name M_640x192_mine \
  --load_weights_folder ../tensorboard/M_640x192_no_IMU/models/weights_29/\
  --eval_split odom_9

# Standard model with rescaling
# python ../evaluate_depth_and_pose.py \
#   --data_path ../kitti_data \
#   --imu_data_path ../kitti_data_unsynced \
#   --model_name M_640x192_no_IMU \
#   --load_weights_folder ../tensorboard/M_640x192_no_IMU/models/weights_29/ \
#   --eval_split raw_odometry

## Standard model without rescaling
#python ../evaluate_depth_and_pose.py --data_path ../kitti_data_unsynced \
#  --model_name M_640x192_no_IMU \
#  --load_weights_folder ../tensorboard/M_640x192_no_IMU/models/weights_X/ \
#  --eval_split raw_odometry \
#  --disable_median_scaling

# Pose2 Without Fuse with rescaling
# python ../evaluate_depth_and_pose.py --data_path ../kitti_data_unsynced \
#   --model_name M_640x192_IMU_2p \
#   --load_weights_folder ../tensorboard/M_640x192_IMU_2p/models/weights_X/ \
#   --eval_split raw_odometry

## Pose2 Without Fuse without rescaling
#python ../evaluate_depth_and_pose.py --data_path ../kitti_data_unsynced \
#  --model_name M_640x192_IMU_2p \
#  --load_weights_folder ../tensorboard/M_640x192_IMU_2p/models/weights_X/ \
#  --eval_split raw_odometry \
#  --disable_median_scaling

# Pose2 With Fuse with rescaling
# python ../evaluate_depth_and_pose.py --data_path ../kitti_data_unsynced \
#   --model_name M_640x192_IMU_2p_pose_fuse \
#   --load_weights_folder ../tensorboard/M_640x192_IMU_2p_pose_fuse/models/weights_X/ \
#   --eval_split raw_odometry

## Pose2 With Fuse without rescaling
#python ../evaluate_depth_and_pose.py --data_path ../kitti_data_unsynced \
#  --model_name M_640x192_IMU_2p_pose_fuse \
#  --load_weights_folder ../tensorboard/M_640x192_IMU_2p_pose_fuse/models/weights_X/ \
#  --eval_split raw_odometry \
#  --disable_median_scaling
