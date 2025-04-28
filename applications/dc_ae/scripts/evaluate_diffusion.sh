export HF_ENDPOINT=https://hf-mirror.com
export MASTER_PORT=29501
export PYTHONPATH=$PYTHONPATH:/data2/user/jyzhang/MIT/efficientvit
cd /data2/user/jyzhang/MIT/efficientvit

# According to benchmark, run f32c32-based model!
# generate reference for FID computation
# torchrun --nnodes=1 --nproc_per_node=8 --master_port 29502 -m applications.dc_ae.generate_reference \
#     dataset=imagenet imagenet.resolution=512 imagenet.image_mean=[0.,0.,0.] imagenet.image_std=[1.,1.,1.] split=train \
#     fid.save_path=assets/data/fid/imagenet_512_train.npz


# Continue to run from here -20250331
# Run evaluation without cfg
# full DC-AE-Diffusion model list: https://huggingface.co/collections/mit-han-lab/dc-ae-diffusion-670dbb8d6b6914cf24c1a49d

# torchrun --nnodes=1 --nproc_per_node=8 --master_port 29502 -m applications.dc_ae.eval_dc_ae_diffusion_model dataset=imagenet_512 model=dc-ae-f64c128-in-1.0-uvit-h-in-512px cfg_scale=1.0 run_dir=tmp
# # Expected results:
# #   fid: 13.754458694549271

# torchrun --nnodes=1 --nproc_per_node=8 --master_port 29503 -m applications.dc_ae.eval_dc_ae_diffusion_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0-usit-h-in-512px cfg_scale=1.0 run_dir=tmp

torchrun --nnodes=1 --nproc_per_node=8 --master_port 29502 -m applications.dc_ae.eval_dc_ae_diffusion_model dataset=imagenet_512 model=dc-ae-f32c32-in-1.0-usit-h-in-512px cfg_scale=1.0 run_dir=tmp
# Expected results:
#   fid: 3.791360749566877

# torchrun --nnodes=1 --nproc_per_node=8 --master_port 29502 -m applications.dc_ae.eval_dc_ae_diffusion_model dataset=imagenet_512 model=/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0-usit-h-in-512px cfg_scale=1.0 run_dir=tmp


# Run evaluation with cfg
# full DC-AE-Diffusion model list: https://huggingface.co/collections/mit-han-lab/dc-ae-diffusion-670dbb8d6b6914cf24c1a49d
# cfg=1.3 for mit-han-lab/dc-ae-f32c32-in-1.0-dit-xl-in-512px
# and cfg=1.5 for all other models
# torchrun --nnodes=1 --nproc_per_node=8 --master_port 29502 -m applications.dc_ae.eval_dc_ae_diffusion_model dataset=imagenet_512 model=dc-ae-f64c128-in-1.0-uvit-h-in-512px cfg_scale=1.5 run_dir=tmp
# # Expected results:
# #   fid: 2.963459255529642

torchrun --nnodes=1 --nproc_per_node=8 --master_port 29502 -m applications.dc_ae.eval_dc_ae_diffusion_model dataset=imagenet_512 model=dc-ae-f32c32-in-1.0-usit-h-in-512px cfg_scale=1.2 run_dir=tmp
# Expected results:
#   fid: 1.883184155951767

torchrun --nnodes=1 --nproc_per_node=8 --master_port 29502 -m applications.dc_ae.eval_dc_ae_diffusion_model dataset=imagenet_512 model=dc-ae-f32c32-in-1.0-usit-2b-in-512px cfg_scale=1.15 run_dir=tmp
# Expected results:
#   fid: 1.7210562991380698

