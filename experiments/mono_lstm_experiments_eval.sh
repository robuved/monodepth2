# Depth evaluation

# aathor trained
# python ../evaluate_depth_and_pose.py \
#   --data_path ../kitti_data \
#   --imu_data_path ../kitti_data_unsynced \
#   --model_name M_640x192_author \
#   --load_weights_folder ../tensorboard/M_640x192_author\
#   --eval_split raw_odometry

# python ../evaluate_depth_and_pose.py \
#   --data_path ../kitti_odom \
#   --model_name M_640x192_mine \
#   --load_weights_folder ../tensorboard/M_640x192_author \
#   --eval_split odom_9

# Standard model with rescaling
# python ../evaluate_depth_and_pose.py \
#   --data_path ../kitti_data \
#   --imu_data_path ../kitti_data_unsynced \
#   --model_name M_640x192_no_IMU \
#   --load_weights_folder ../tensorboard/M_640x192_no_IMU/models/weights_29/ \
#   --eval_split raw_odometry


# Pose2 With Fuse with rescaling
python ../evaluate_depth_and_pose.py \
  --data_path ../kitti_data \
  --imu_data_path ../kitti_data_unsynced \
  --model_name M_640x192_IMU_2p_pose_fuse \
  --load_weights_folder ../tensorboard/M_640x192_IMU_2p_pose_fuse/models/weights_10 \
  --eval_split raw_odometry \
  --use_imu