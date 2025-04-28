export HF_ENDPOINT=https://hf-mirror.com
cd /data2/user/jyzhang/MIT/efficientvit
export MASTER_PORT=29505
export PYTHONPATH=$PYTHONPATH:/data2/user/jyzhang/MIT/efficientvit


# Training Code for dc-ae-f32c32-in-1.0-w4-v25
# Expected FID: 2.22879, PSNR: 26.22011, SSIM: 0.72431, LPIPS: 0.12579
# Based on model w4-v3, decoder.depth_list=[0,5,10,2,2,2] -> [0,1,2,1,1,2], 
# Compression Ratio 12%, 40% reduction in total MACs, 65% reduction in decoder MACs
CUDA_VISIBLE_DEVICES=7 python /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/scripts/dc_de_distillation_gan.py \
--batch_size 4 --learning_rate_G 1e-4 --learning_rate_D 1e-4 --num_epochs 15 --train_samples 600 \
--student_model_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w4-v3 \
--model_save_dir /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w4-v25 \
--pic_save_dir /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/pic_results_w4_v25 \
--alpha_disti 100 --alpha_img 1 --beta 0.1 --gamma 0.3 --gan_ratio 300 --align 3 --freeze_proj_out True --freeze_encoder True \
--cosine_T_0_G 5 --cosine_T_mult_G 1 --eta_min_G 1e-6 --weight_decay_G 0.01 \
--cosine_T_0_D 5 --cosine_T_mult_D 1 --eta_min_D 1e-6 --weight_decay_D 0.01 \
--dynamic_loss True --division_epoch 10 \
--accumulate_batch False --accumulation_steps 4 \
--shallow_train False --shallow_training_epochs 5 --model_config dc-ae-f32c32-in-1.0-pruned-w4-v3

CUDA_VISIBLE_DEVICES=7 torchrun --nnodes=1 --nproc_per_node=1 --master_port 29505 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w4-v25 run_dir=tmp


# Training Code for dc-ae-f32c32-in-1.0-w4-v8
# Expected FID: 1.69890, PSNR: 26.55881, SSIM: 0.73866, LPIPS: 0.11312
# Based on model w4-v1, decoder.depth_list=[0,5,10,2,2,2] -> [0,3,5,2,2,2], 
# Compression Ratio 8%, 22% reduction in total MACs and 40% reduction in decoder MACs
CUDA_VISIBLE_DEVICES=7 python /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/scripts/dc_de_distillation_gan.py \
--batch_size 4 --learning_rate_G 1e-4 --learning_rate_D 1e-4 --num_epochs 15 --train_samples 600 \
--student_model_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w4-v1 \
--model_save_dir /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w4-v8 \
--pic_save_dir /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/pic_results_w4_v8 \
--alpha_disti 100 --alpha_img 1 --beta 0.1 --gamma 0.1 --gan_ratio 300 --align 3 --freeze_proj_out True --freeze_encoder True \
--cosine_T_0_G 5 --cosine_T_mult_G 1 --eta_min_G 1e-6 --weight_decay_G 0.01 \
--cosine_T_0_D 5 --cosine_T_mult_D 1 --eta_min_D 1e-6 --weight_decay_D 0.01 \
--dynamic_loss False --division_epoch 10 \
--accumulate_batch False --accumulation_steps 4 \
--shallow_train False --shallow_training_epochs 5 --model_config dc-ae-f32c32-in-1.0-pruned-w4-v1

CUDA_VISIBLE_DEVICES=7 torchrun --nnodes=1 --nproc_per_node=1 --master_port 29505 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w4-v8 run_dir=tmp


# Training Code for dc-ae-f32c32-in-1.0-w4-v4
# Expected FID:0.83769, PSNR: 26.5646, SSIM: 0.73160, LPIPS: 0.09752
# Based on model w3-v2, decoder.depth_list=[0,5,10,2,2,2] -> [0,5,10,1,1,2], 
# Compression Ratio 10%, 1.5% reduction in total MACs
CUDA_VISIBLE_DEVICES=7 python /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/scripts/dc_de_distillation_gan.py \
--batch_size 4 --learning_rate_G 1e-4 --learning_rate_D 1e-4 --num_epochs 15 --train_samples 600 \
--student_model_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w3-v2 \
--model_save_dir /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w4-v4 \
--pic_save_dir /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/pic_results_w4_v4 \
--alpha_disti 100 --alpha_img 1 --beta 0.1 --gamma 0.05 --gan_ratio 300 --align 3 --freeze_proj_out True --freeze_encoder True \
--cosine_T_0_G 5 --cosine_T_mult_G 1 --eta_min_G 1e-6 --weight_decay_G 0.01 \
--cosine_T_0_D 5 --cosine_T_mult_D 1 --eta_min_D 1e-6 --weight_decay_D 0.01 \
--dynamic_loss False --division_epoch 10 \
--accumulate_batch False --accumulation_steps 4 \
--shallow_train False --shallow_training_epochs 5 --model_config dc-ae-f32c32-in-1.0-pruned-w3-v2

CUDA_VISIBLE_DEVICES=7 torchrun --nnodes=1 --nproc_per_node=1 --master_port 29505 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w4-v4 run_dir=tmp




