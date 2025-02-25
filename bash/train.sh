CUDA_VISIBLE_DEVICES=0 python train.py --bf16 True --eval_steps 100 --save_dir 256img_headnum1_depth11111 --data_name 256 --per_device_train_batch_size 2 
CUDA_VISIBLE_DEVICES=1 python train.py --bf16 True --eval_steps 100 --save_dir 512img_headnum1_depth11111 --data_name 512 --per_device_train_batch_size 1

CUDA_LAUNCH_BLOCKING=1 CUDA_VISBLE_DEVICES=0 python train.py --bf16 True --eval_steps 100 --save_dir 256img_headnum1_depth11111_resize_focalloss_high_freq_ratio10_fix_the_model --data_name 256 --per_device_train_batch_size 2 --high_freq_ratio 10
CUDA_VISBLE_DEVICES=0 python train.py --bf16 True --eval_steps 100 --save_dir 2_11_5_44 --data_name 256 --per_device_train_batch_size 2 --high_freq_ratio 10