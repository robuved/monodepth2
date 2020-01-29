# Our standard mono model
python ../train.py --data_path ./kitty_data_uncynced --model_name M_640x192 \
  --log_dir ./tensorboard/ --split raw_odometry --use_imu --frame_ids 0 -2 -1 1 2

# python ../train.py --data_path ./kitty_data_uncynced --model_name M_640x192 \
#   --log_dir ./tensorboard/ --split raw_odometry --use_imu --frame_ids 0 -2 -1 1 2
