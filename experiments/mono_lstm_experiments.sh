# Our standard mono model

# python ../train.py --data_path ../kitti_data_unsynced --model_name M_640x192_no_IMU \
#   --log_dir ../tensorboard/ --split raw_odometry --frame_ids 0 -2 -1 1 2 \
#   --batch_size 6 --num_epochs 30


python ../train.py --data_path ../kitti_data_unsynced --model_name M_640x192_IMU_2p_pose_fuse \
  --log_dir ../tensorboard/ --split raw_odometry --use_imu --frame_ids 0 -1 -2 1 2 \
  --batch_size 6 --num_epochs 30 --pose_fuse

# python ../train.py --data_path ./kitty_data_unsynced --model_name M_640x192 \
#   --log_dir ./tensorboard/ --split raw_odometry --use_imu --frame_ids 0 -2 -1 1 2


# python -m torch.utils.bottleneck ../train.py --data_path ../kitti_data_unsynced --model_name M_640x192_no_IMU \
#   --log_dir ../tensorboard/ --split raw_odometry_small --frame_ids 0 -1 -2 1 2 \
#   --batch_size 6 --log_frequency 1 --num_epochs 1 --num_workers 0

# python -m torch.utils.bottleneck ../train.py --data_path ../kitti_data_unsynced --model_name M_640x192_IMU \
#   --log_dir ../tensorboard/ --split raw_odometry_small --use_imu --frame_ids 0 -1 -2 1 2 \
#   --batch_size 6 --log_frequency 1 --num_epochs 1