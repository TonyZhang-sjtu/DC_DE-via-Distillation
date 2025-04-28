export HF_ENDPOINT=https://hf-mirror.com
cd /data2/user/jyzhang/MIT/efficientvit

# CUDA_VISIBLE_DEVICES=0 python /home/jyzhang/data2/MIT/efficientvit/applications/dc_ae/scripts/test_speed.py \
# --model_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0 \
# --n_iters 1000 --batch_size 1 --target both


# CUDA_VISIBLE_DEVICES=0 python /home/jyzhang/data2/MIT/efficientvit/applications/dc_ae/scripts/test_speed.py \
# --model_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0 \
# --n_iters 1000 --batch_size 1 --target decoder

# CUDA_VISIBLE_DEVICES=0 python /home/jyzhang/data2/MIT/efficientvit/applications/dc_ae/scripts/test_speed.py \
# --model_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w4-v1 \
# --n_iters 1000 --batch_size 1 --target both

# CUDA_VISIBLE_DEVICES=0 python /home/jyzhang/data2/MIT/efficientvit/applications/dc_ae/scripts/test_speed.py \
# --model_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w4-v1 \
# --n_iters 1000 --batch_size 1 --target decoder


# CUDA_VISIBLE_DEVICES=0 python /home/jyzhang/data2/MIT/efficientvit/applications/dc_ae/scripts/test_speed.py \
# --model_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w4-v2 \
# --n_iters 1000 --batch_size 1 --target both

# CUDA_VISIBLE_DEVICES=0 python /home/jyzhang/data2/MIT/efficientvit/applications/dc_ae/scripts/test_speed.py \
# --model_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w4-v2 \
# --n_iters 1000 --batch_size 1 --target decoder


CUDA_VISIBLE_DEVICES=0 python /home/jyzhang/data2/MIT/efficientvit/applications/dc_ae/scripts/test_speed.py \
--model_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w4-v3 \
--n_iters 1000 --batch_size 1 --target both

CUDA_VISIBLE_DEVICES=0 python /home/jyzhang/data2/MIT/efficientvit/applications/dc_ae/scripts/test_speed.py \
--model_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w4-v3 \
--n_iters 1000 --batch_size 1 --target decoder
