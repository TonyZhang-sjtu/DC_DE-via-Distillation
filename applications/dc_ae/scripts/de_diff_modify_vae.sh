export HF_ENDPOINT=https://hf-mirror.com
cd /data2/user/jyzhang/MIT/efficientvit

python /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/scripts/de_diff_modify_vae.py \
--encoder_model /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v7 \
--diffusion_model /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0-usit-h-in-512px \
--save_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-usit-h-in-512px-v7/model.safetensors \
--prune_method direct

