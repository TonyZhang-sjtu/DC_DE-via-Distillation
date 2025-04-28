export HF_ENDPOINT=https://hf-mirror.com
export MASTER_PORT=29501
export PYTHONPATH=$PYTHONPATH:/data2/user/jyzhang/MIT/efficientvit
cd /data2/user/jyzhang/MIT/efficientvit

# Generate reference for FID computation:
# torchrun --nnodes=1 --nproc_per_node=8 --master_port 29501 -m applications.dc_ae.generate_reference \
#     dataset=imagenet imagenet.resolution=512 imagenet.image_mean=[0.,0.,0.] imagenet.image_std=[1.,1.,1.] split=test \
#     fid.save_path=assets/data/fid/imagenet_512_val.npz


# Run evaluation:
# full DC-AE model list: https://huggingface.co/collections/mit-han-lab/dc-ae-670085b9400ad7197bb1009b
# torchrun --nnodes=1 --nproc_per_node=8 --master_port 29501 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=dc-ae-f64c128-in-1.0 run_dir=tmp

# Reproduce Correctly!
# Expected results:
#   fid: 0.2167766520628902
#   psnr: 26.1489275
#   ssim: 0.710486114025116
#   lpips: 0.0802311897277832

# f32c32 benchmark
# torchrun --nnodes=1 --nproc_per_node=8 --master_port 29501 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=dc-ae-f32c32-in-1.0 run_dir=tmp


# # 0401_v1
# torchrun --nnodes=1 --nproc_per_node=1 --master_port 29501 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v8 run_dir=tmp

# # 0401_v6
# torchrun --nnodes=1 --nproc_per_node=1 --master_port 29501 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v9 run_dir=tmp

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 --master_port 29503 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w2-v14 run_dir=tmp

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 --master_port 29503 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w2-v9 run_dir=tmp

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 --master_port 29503 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w2-v10 run_dir=tmp

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 --master_port 29503 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w2-v11 run_dir=tmp

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 --master_port 29503 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w2-v12 run_dir=tmp

# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 --master_port 29503 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w2-v6 run_dir=tmp

# torchrun --nnodes=1 --nproc_per_node=1 --master_port 29501 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v8-2 run_dir=tmp

# # 0401_v7
# torchrun --nnodes=1 --nproc_per_node=1 --master_port 29501 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v10 run_dir=tmp

# torchrun --nnodes=1 --nproc_per_node=1 --master_port 29501 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v11 run_dir=tmp

# torchrun --nnodes=1 --nproc_per_node=1 --master_port 29501 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v12 run_dir=tmp

# torchrun --nnodes=1 --nproc_per_node=1 --master_port 29501 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v13 run_dir=tmp

# torchrun --nnodes=1 --nproc_per_node=1 --master_port 29501 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v14 run_dir=tmp

# torchrun --nnodes=1 --nproc_per_node=1 --master_port 29501 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v15 run_dir=tmp

# torchrun --nnodes=1 --nproc_per_node=1 --master_port 29501 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v14-2 run_dir=tmp

# torchrun --nnodes=1 --nproc_per_node=8 --master_port 29501 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v10 run_dir=tmp



export HF_ENDPOINT=https://hf-mirror.com
cd /data2/user/jyzhang/MIT/efficientvit

# CUDA_VISIBLE_DEVICES=0 python /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/scripts/demo-dc-ae-recons.py

# CUDA_VISIBLE_DEVICES=0 python /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/scripts/dc_de_distillation.py \
# --batch_size 1 --learning_rate 1e-4 --num_epochs 10 --train_samples 10 --batch_size 1 \
# --model_save_dir /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/0401-v1

# for occupying GPU cards!

# L1+LPIPS 0403 v15
CUDA_VISIBLE_DEVICES=1 python /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/scripts/dc_de_distillation_nto1.py \
--batch_size 1 --learning_rate 1e-4 --num_epochs 5000 --train_samples 1000 --batch_size 1 \
--student_model_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v3-wo \
--model_save_dir /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v15-2 \
--pic_save_dir /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/pic_results_v15-2 \
--alpha 1.0 --beta 0.1 --gamma 0.05 --gan_ratio 25


CUDA_VISIBLE_DEVICES=1 python /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/scripts/dc_de_distillation_nto1.py \
--batch_size 1 --learning_rate 1e-4 --num_epochs 5000 --train_samples 1000 --batch_size 1 \
--student_model_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v3-wo \
--model_save_dir /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v15-2 \
--pic_save_dir /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/pic_results_v15-2 \
--alpha 1.0 --beta 0.1 --gamma 0.05 --gan_ratio 25


CUDA_VISIBLE_DEVICES=1 python /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/scripts/dc_de_distillation_nto1.py \
--batch_size 1 --learning_rate 1e-4 --num_epochs 5000 --train_samples 1000 --batch_size 1 \
--student_model_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v3-wo \
--model_save_dir /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v15-2 \
--pic_save_dir /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/pic_results_v15-2 \
--alpha 1.0 --beta 0.1 --gamma 0.05 --gan_ratio 25





